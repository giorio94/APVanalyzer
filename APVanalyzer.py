"""
This module provides a set of classes allowing to analyze an HAR capture representing the data exchanged during a
playback of Amazon Prime Video media. It mainly provides the possibility to plot different graphs concerning,
for example, download throughput, selected video bitrate, estimated buffer sizes and many more.
Along to same helpers, the main implemented classes are:

- ShaperCSVParser: reads and parses the shaper file, a csv file containing the log of how the traffic shaper has been
  configured.
- MPDParser: parses an MPD file and exposes the available video representations (i.e. different available bitrate)
  therein present.
- HARAnalyzer: reads and parses the content of an HAR file referred to an Amazon Prime Video playback, and provides a
  number of methods to plot summary graphs.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse
import xml.etree.ElementTree as ElementTree
from collections import defaultdict
from typing import Iterable, List, Tuple, Union

import dateutil.parser
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


class APVanalyzerException(Exception):
    """
    This class is used to represent exceptions related to APVanalyzer.
    """


class TimeArrayHelper:
    """
    This utility class is used to wrap a numpy array so that each element corresponds to one second of a video.
    In particular, it provides a method to uniformly add a value (e.g. an amount of bytes) between a begin and end time.
    """

    def __init__(self, total_duration_ms: int):
        """
        Initialises the numpy array with a length corresponding to the specified duration.
        :param total_duration_ms: the total duration to be considered (in milliseconds).
        """

        _array_length = int(np.ceil(total_duration_ms / 1000))
        self.total_duration_ms = total_duration_ms
        self.array = np.zeros(_array_length)

    def average(self):
        """
        Returns the average value computed as the sum of the array content divided by the total number of seconds.
        :return: the computed average value.
        """
        return np.sum(self.array) / self.total_duration_ms * 1000

    def increment(self, start_ms: int, end_ms: int, value: float):
        """
        Uniformly increments all the affected array cells according to the specified parameters, with a per-millisecond
        increment computed as "value / (end_time - start_time)".
        :param start_ms: the initial millisecond to be considered.
        :param end_ms: the last millisecond to be considered.
        :param value: the value to be uniformly added.
        """

        _total_ms = end_ms - start_ms
        _init_idx = np.floor(start_ms / 1000).astype(int)
        _last_idx = np.floor(end_ms / 1000).astype(int)
        _diff_idx = _last_idx - _init_idx

        # Initial second
        _init_percentage = 1.0 if _diff_idx == 0 else (1000 - start_ms % 1000) / _total_ms
        self.array[_init_idx] += value * _init_percentage

        # The value spans over two or more seconds
        if _diff_idx > 0:
            _last_percentage = (end_ms % 1000) / _total_ms
            self.array[_last_idx] += value * _last_percentage

            # The value spans over more than two seconds
            if _diff_idx > 1:
                _remaining_percentage = (1 - _init_percentage - _last_percentage) / (_diff_idx - 1)
                self.array[_init_idx + 1:_last_idx] += value * _remaining_percentage


class BufferHelper:
    """
    This utility class is used to wrap a numpy array representing, for each element, the amount of data available
    int the buffer during that second.
    """

    def __init__(self, total_duration_ms: int, buffer_flushed):
        """
        Initialises the numpy array with a length corresponding to the specified duration.
        :param total_duration_ms: the total duration to be considered (in milliseconds).
        :param buffer_flushed: a sorted array of time instants when the buffer was flushed (in milliseconds).
        """

        _buffer_length = int(np.ceil(total_duration_ms / 1000))
        self.total_duration_ms = total_duration_ms
        self.buffer = np.zeros(_buffer_length)
        self.buffer_flushed = np.append(buffer_flushed, total_duration_ms)

    def data_received(self, timestamp: int, received_ms: int):
        """
        Increments the amount of buffered data from the specified timestamp.
        :param timestamp: the time instant when the data has been deceived (in milliseconds).
        :param received_ms: the amount of received data (milliseconds of video).
        """

        if timestamp > self.total_duration_ms:
            raise ValueError("Invalid timestamp value")

        _init_idx = np.floor(timestamp / 1000).astype(int)
        _flush_idx = np.ceil(self.buffer_flushed[self.buffer_flushed >= timestamp][0] / 1000).astype(int)

        self.buffer[_init_idx:_flush_idx] += received_ms

    def video_playback(self, start_ms: int, stop_ms: int):
        """
        Decrements the amount of buffered data, considering that the video has been playback in the specified interval.
        :param start_ms: the initial playback instant (in milliseconds).
        :param stop_ms: the final playback instant (in milliseconds).
        """

        _start_idx = np.floor(start_ms / 1000).astype(int)
        _end_idx = np.floor(stop_ms / 1000).astype(int)
        _flush_idx = np.ceil(self.buffer_flushed[self.buffer_flushed >= stop_ms][0] / 1000).astype(int)

        if _start_idx == _end_idx:
            self.buffer[_start_idx:] -= stop_ms - start_ms
            return

        _decrement = [(1000 - start_ms % 1000) + _i * 1000 for _i in range(_end_idx - _start_idx)]
        final_decrement = _decrement[-1] + stop_ms % 1000

        self.buffer[_start_idx:_end_idx] -= _decrement
        self.buffer[_end_idx:_flush_idx] -= final_decrement


class IntervalsHelper:
    """
    This utility class wraps some utility methods used to operate on intervals.
    """

    @staticmethod
    def merge_intervals(intervals: Iterable[Tuple[int, int]]):
        """
        Merges the intervals received in input, so that the output contains only non-overlapping intervals.
        Algorithm slightly adapted from https://codereview.stackexchange.com/a/108651
        :param intervals: an array of tuples representing the intervals to be merged (assumed to be sorted by lower
        bound).
        :return: an iterator that computes the merged intervals.
        """

        try:
            # _low and _high represent the bounds of the current run of merges
            _low, _high = next(iter(intervals))
        except StopIteration:
            return  # Nothing to merge

        for _new_low, _new_high in intervals:

            if _new_low <= _high:  # The new interval overlaps the current run
                _high = max(_high, _new_high)  # Merge the two overlapping intervals

            else:  # The current run is over
                yield _low, _high  # Yield the accumulated interval
                _low, _high = _new_low, _new_high  # Start a new run

        yield _low, _high  # Terminate the final run

    @staticmethod
    def opposite_intervals(intervals: Iterable[Tuple[int, int]], start: int, end: int):
        """
        Generates the opposite intervals with respect to the ones received in input (i.e. representing the empty spaces
        in the original intervals).
        :param intervals: an array of tuples representing the intervals to be considered (assumed to be merged and
        sorted by lower bound).
        :param start: the lower bound for the first interval.
        :param end: the higher bound for the last interval.
        :return: an iterator that computes the opposite intervals.
        """

        for _low, _high in intervals:
            if start < _low:  # Avoid problems in case the first lower bound is to the left of start
                yield start, _low
            start = _high

        if start < end:  # Avoid problems in case the last higher bound is to the right of end
            yield start, end


class ShaperCSVParser:
    """
    This class reads and parses the shaper file, a csv file containing the log of how the traffic shaper has been
    configured.
    """

    def __init__(self, path: str, filename: str):
        """
        Initializes the object by parsing the content of the shaper file (in case it is not found, a warning is issued).
        :param path: the path where the shaper file is stored.
        :param filename: the name of the shaper file.
        """

        _shaper_file_path = os.path.join(path, filename)

        try:
            self.times, self.download_rates = np.loadtxt(_shaper_file_path, dtype=int, unpack=True, ndmin=2)
            sys.stdout.write("Shaper file correctly read: {} configurations".format(len(self.times)))
        except IOError as _ex:
            self.times, self.download_rates = None, None
            sys.stderr.write("Failed reading shaper file ({}): {}\n".format(_shaper_file_path, str(_ex)))

    def plot_shaper_rates(self, axis, initial_time_ms: int, total_time_ms: int, **kwargs):
        """
        Plot the graph representing the variations in the shaper download rates.
        :param axis: the axis where the graph is plot to.
        :param initial_time_ms: the initial time instant (in milliseconds).
        :param total_time_ms: the total elapsed time (in milliseconds).
        :param kwargs: additional parameters forwarded to the plot method (e.g. line color, width, label, ...).
        :return: the object representing the plotted line (or None in case no information are available.
        """

        if not self.shaper_file_loaded():
            return None

        _times = np.append(self.times - initial_time_ms, total_time_ms)
        _rates = np.append(self.download_rates, self.download_rates[-1])
        return axis.step(_times / 1000, _rates / 1e6, where="post", **kwargs)

    def shaper_file_loaded(self):
        """
        Returns whether the shaper file content has been correctly loaded.
        :return: a boolean value indicating the outcome of the loading operation.
        """

        return self.times is not None


class MPDParser:
    """
    This class parses an MPD file and exposes the available video representations (i.e. different available bitrate)
    therein present.
    """

    def __init__(self, path: str, filename: str):
        """
        Initialises the object by parsing and storing the content of the MPD file.
        :param path: the path where the MPD file is stored.
        :param filename: the name of the MPD file.
        """

        self.filename, _ = os.path.splitext(filename)
        _mpd_representation = MPDParser.__load_mpd_file(path, filename)
        _video_adaptation_set = MPDParser.__get_video_adaptation_set(_mpd_representation)

        self._chunks = MPDParser.__extract_video_chunks(_video_adaptation_set)
        self._representations = MPDParser.__extract_video_representations(_video_adaptation_set)
        self.__build_mpd_bitrate_information()

    @staticmethod
    def download_mpd_file(mpd_urls: List[str], path: str, filename: str):
        """
        Tries to download an MPD file from the list of given URLs and, in case of success, returns a MPDParser
        representing it.
        :param mpd_urls: the list of URLs from where download the MPD file.
        :param path: the path where the MPD file is going to be saved.
        :param filename: the name assigned to the downloaded MPD file.
        :return: an MPDParser instance representing the downloaded object.
        """

        _mpd_file_path = os.path.join(path, filename)
        for _url in mpd_urls:
            sys.stdout.write("Trying to download MPD file from '{}'...\n".format(_url))

            try:
                urllib.request.urlretrieve(_url, filename=_mpd_file_path)
                sys.stdout.write("MPD file correctly downloaded and saved as '{}'\n".format(_mpd_file_path))
                return MPDParser(path, filename)
            except urllib.error.URLError as _ex:
                sys.stderr.write("Failed to download MPD file: {}\n".format(str(_ex)))

        raise APVanalyzerException("Impossible to download the MPD file")

    def get_video_duration(self):
        """
        Returns the total video duration.
        :return: the video duration (in milliseconds).
        """

        return self._chunks[-1]["end-playback-ms"]

    def get_video_chunks_number(self):
        """
        Returns the total number of chunks the video is divided in.
        :return: the total number of chunks the video is divided in.
        """

        return len(self._chunks)

    def get_nominal_bitrate(self, quality: int):
        """
        Given the quality of a representation, returns the associated nominal bitrate.
        :param quality: the quality of the selected representation.
        :return: the nominal bitrate associated to the selected representation.
        """

        return self._representations[quality - 1]["nominal-bitrate"]

    def lookup_chunk(self, quality: int, byte_range: Tuple[int, int]):
        """
        Given the quality of a representation and a byte-range, looks for the information about that video chunk.
        :param quality: the quality of the selected representation.
        :param byte_range: the downloaded byte-range of the specific representation.
        :return: a tuple containing the index of the chunk and the information about the chunk itself.
        """

        _segments = self._representations[quality - 1]["segments"]
        _array_idx = list(map(lambda _segment: _segment["byte-range"], _segments)).index(byte_range)
        return _array_idx, self._chunks[_array_idx]

    def plot_representation_bitrate_plots(self, title=None, figsize=None, nrows=3):
        """
        Plots the graph of the bitrate (nominal, average and actual) associated to each representation.
        :param title: the title of the plot.
        :param figsize: the size of the created figure.
        :param nrows: the number of rowa to be created in the plot.
        """

        if title is None:
            title = "{} (MPD Representations)".format(self.filename)

        _ncols = np.floor(len(self._representations) / nrows).astype(int)
        _f, _ax_matrix = plt.subplots(nrows, _ncols, sharex="all", num=title, figsize=figsize, squeeze=False)

        _f.suptitle(title, fontsize="x-large")
        for _idx, _ax in enumerate(_ax_matrix.flat):
            self.plot_representation_bitrate_plot(_idx, _ax, add_label=_idx == 0)
        _f.legend(loc="lower center", ncol=3)
        _f.tight_layout()

    def plot_representation_bitrate_plot(self, representation_idx, axis=None, add_label=True):
        """
        Plots the graph of the bitrate (nominal, average and actual) associated to a specific representation.
        :param representation_idx: the index of the representation to be plot.
        :param axis: the axis where the graph is plot.
        :param add_label: whether a label is added to the plots (to avoid cluttering the legend)
        """

        _repr = self._representations[representation_idx]
        _duration = self._chunks[-1]["end-playback-ms"] / 1000

        if axis is None:
            _, axis = plt.subplots(1, 1)

        axis.plot(list(map(lambda _bitrate: _bitrate / 1e6, _repr["actual-bitrate"])),
                  label="Actual bitrate" if add_label else None, linewidth=0.5)
        axis.axhline(_repr["average-bitrate"] / 1e6, xmax=_duration,
                     label="Average bitrate" if add_label else None, linewidth=2, color="C2")
        axis.axhline(_repr["nominal-bitrate"] / 1e6, xmax=_duration,
                     label="Nominal bitrate" if add_label else None, linewidth=2, color="C3")

        axis.set_title("{} Mbps".format(_repr["nominal-bitrate"] / 1e6))
        axis.set_xlabel("Playback time (s)")
        axis.set_ylabel("Bitrate (Mbps)")

    @staticmethod
    def __load_mpd_file(path: str, filename: str):
        """
        Loads the content of the MPD file.
        :param path: the path where the MPD file is stored.
        :param filename: the name of the MPD file.
        :return: the in-memory representation of the MPD file (ElementTree).
        """

        _mpd_file_path = os.path.join(path, filename)
        with open(_mpd_file_path, "r") as _file:
            return ElementTree.parse(_file).getroot()

    @staticmethod
    def __get_video_adaptation_set(mpd_representation):
        """
        Extracts the AdaptationSet associated to the video chunks.
        :param mpd_representation: the in-memory representation of the MPD file.
        :return: the in-memory representation of the AdaptationSet associated to the video chunks.
        """

        _urn_adaptation_set = "./{urn:mpeg:dash:schema:mpd:2011}AdaptationSet[@contentType='video']"
        _video_adaptation_set = mpd_representation[0].find(_urn_adaptation_set)

        return _video_adaptation_set

    @staticmethod
    def __extract_video_chunks(video_adaptation_set):
        """
        Extracts, for each video chunk, its duration and the associated begin and end playback times.
        :param video_adaptation_set: the in-memory representation of the video AdaptationSet element of the MPD file.
        :return: a numpy array of dictionaries containing the extracted values
        """

        # The urns used to identify the XML elements of interest
        _urn_encoded_segment_duration = "{urn:mpeg:dash:schema:mpd:2011}EncodedSegmentDurations"

        _duration_element = video_adaptation_set.find(_urn_encoded_segment_duration)
        _timescale = int(_duration_element.attrib["timescale"])

        # Split the duration values from the original string
        _durations_map = map(lambda _d: int(_d, 16) / _timescale * 1000, _duration_element.text[:-1].split(";"))
        # Add an initial zero element since the first chunk contains only metadata
        _durations = np.concatenate(([0.0, ], list(_durations_map)))

        # Compute the begin and end playback times
        _durations_cumsum = np.cumsum(_durations)
        _begin_playback_times = np.concatenate(([0.0, ], _durations_cumsum[:-1]))
        _end_playback_times = _durations_cumsum

        return np.array([{
            "duration": _d,
            "begin-playback-ms": _b,
            "end-playback-ms": _e
        } for _d, _b, _e in zip(_durations, _begin_playback_times, _end_playback_times)])

    @staticmethod
    def __extract_video_representations(video_adaptation_set):
        """
        Extracts, for each available video representation, its bitrate and the associated list of segments (initial and
        final byte number of each chunk and corresponding length).
        :param video_adaptation_set: the in-memory representation of the video AdaptationSet element of the MPD file.
        :return: a numpy array of dictionaries containing te extracted values.
        """

        _urn_representation = "{urn:mpeg:dash:schema:mpd:2011}Representation"
        _urn_encoded_segment_list = "{urn:mpeg:dash:schema:mpd:2011}EncodedSegmentList"

        _representations = []
        for _representation in video_adaptation_set.iter(_urn_representation):
            _segments = _representation.find(_urn_encoded_segment_list).text[:-1].split(";")
            _segments_begin = np.array(list(map(lambda _s: int(_s.split("-")[0], 16), _segments)))
            _segments_end = np.array(list(map(lambda _s: int(_s.split("-")[1], 16), _segments)))
            _segments_length = _segments_end - _segments_begin

            _representations.append({
                "nominal-bitrate": int(_representation.attrib["bandwidth"]),
                "segments": np.array([{
                    "byte-range": (_b, _e),
                    "byte-length": _l,
                } for _b, _e, _l in zip(_segments_begin, _segments_end, _segments_length)])
            })

        return _representations

    def __build_mpd_bitrate_information(self):
        """
        Computes, for each available video representation, its actual (i.e. per-second) and average bitrate.
        """
        # For each representation, compute the actual bitrate for every playback second
        for _representation in self._representations:

            _actual_bitrate = TimeArrayHelper(self.get_video_duration())
            # Skipping the first element, since it is the header
            for _chunk, _segment in zip(self._chunks[1:], _representation["segments"][1:]):
                # Note: bitrate computed in bits/second (thus the *8 multiplication)
                _actual_bitrate.increment(_chunk["begin-playback-ms"], _chunk["end-playback-ms"],
                                          _segment["byte-length"] * 8)

            _representation["actual-bitrate"] = _actual_bitrate.array
            _representation["average-bitrate"] = _actual_bitrate.average()


class HARAnalyzer:
    """
    This class reads and parses the content of an HAR file referred to an Amazon Prime Video playback, and provides a
    number of methods to plot summary graphs.
    """

    def __init__(self, path: str, filename: str, mpd_parser: Union[MPDParser, None],
                 shaper_parser: Union[ShaperCSVParser, None]):
        """
        Initialises the object by parsing and storing the content of the HAR file.
        :param path: the path where the HAR file is stored.
        :param filename: the name of the HAR file.
        :param mpd_parser: the object representing an MPD file (if None, this function tries to
        download the file by considering the URLs referred to in the HAR file).
        :param shaper_parser: the object containing the information about the shaper rates.
        """

        self.filename, _ = os.path.splitext(filename)

        # Save the additional parsers received as parameters
        self.mpd_parser = mpd_parser
        self.shaper_parser = shaper_parser

        # Load the information from the HAR file
        _har_representation = HARAnalyzer.__load_har_file(path, filename)

        # Extract the information about the downloaded chunks of media data
        self.downloaded_chunks = HARAnalyzer.__extract_downloaded_chunks_information(_har_representation)
        self.initial_download_time_ms = HARAnalyzer.__compute_relative_times_information(self.downloaded_chunks)
        self.total_download_time_ms = HARAnalyzer.__get_total_download_time(self.downloaded_chunks)
        self.download_throughput = HARAnalyzer.__compute_download_throughput(
            self.downloaded_chunks, self.total_download_time_ms)

        # Extract the information about various download and playback events
        self.events_info = HARAnalyzer.__extract_events(
            _har_representation, self.initial_download_time_ms, self.total_download_time_ms)
        self.events_info["cdn-changes"] = HARAnalyzer.__extract_cdn_changes(self.downloaded_chunks)

        # Try to download the MPD file if not already loaded
        self.mpd_urls = HARAnalyzer.__extract_mpd_urls(_har_representation)
        if self.mpd_parser is None:
            # An exception is raised if it is not possible to download the file
            # (required to continue the initialization)
            _mpd_filename = "{}.mpd".format(self.filename)
            self.mpd_parser = MPDParser.download_mpd_file(self.mpd_urls, path, _mpd_filename)

        self.playback_bitrates = HARAnalyzer.__compute_playback_bitrates(self.downloaded_chunks, self.mpd_parser)
        self.buffered_data = HARAnalyzer.__compute_buffered_data(self.downloaded_chunks, self.events_info,
                                                                 self.mpd_parser, self.total_download_time_ms)

    def plot_download_timescale_graph(self, title=None, figsize=None):
        """
        Plots in a single figure the three graphs in download timescale.
        :param title: the title associated to the graph (if none, it is used the HAR filename, without extension).
        :param figsize: the size of the created figure.
        """

        if title is None:
            title = "{} (Download Timescale)".format(self.filename)

        _f, _axis = plt.subplots(3, 1, sharex="all", num=title, gridspec_kw={"height_ratios": [8, 2, 1]}, figsize=figsize)
        _f.suptitle(title)

        self.plot_throughput_bitrate_graph(axis=_axis[0], title="Throughput and Bitrate", interactive_toggle_lines=True)
        self.plot_buffer_size_graph(axis=_axis[1], title="Buffer size", references=[0, 60, 120], accessory_legend=False)
        self.plot_requests_on_off_graph(axis=_axis[2], title="Requests On/Off", accessory_legend=False)

    def plot_throughput_bitrate_graph(self, title, axis=None, accessory_legend=True, interactive_toggle_lines=False):
        """
        Plot the graph representing the variations in download throughput and bitrates.
        :param title: the title associated to the plotted graph.
        :param axis: the axis where the graph is plot to (if None, a new one is created).
        :param accessory_legend: whether the accessory lines are shown in the legend.
        :param interactive_toggle_lines: whether the graph is configured to allow interactively toggling the lines
        visibility.
        """

        if axis is None:
            _, axis = plt.subplots(1, 1, num=title)

        axis.set_title(title)

        _lines = [
            axis.plot(self.download_throughput / 1e6, label="Download throughput", linewidth=1, zorder=0),
            self._plot_requested_bitrate(axis, label="Requested bitrate", zorder=1, color="C2"),
            self._plot_playback_bitrate(axis, label="Playback bitrate", zorder=-5, color="C3"),
            self.shaper_parser.plot_shaper_rates(axis, self.initial_download_time_ms, self.total_download_time_ms,
                                                 label="Traffic shaper", color="C4") if self.shaper_parser else None,
        ]

        self._plot_download_timescale_accessory_lines(axis, accessory_legend)

        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Throughput/Bitrate (Mbps)")

        # Reorder the legend entries (otherwise the LineCollection is the last)
        _handles, _labels = axis.get_legend_handles_labels()
        _handles.insert(1, _handles.pop(-1))
        _labels.insert(1, _labels.pop(-1))
        _legend_lines = axis.legend(_handles, _labels).get_lines()

        if interactive_toggle_lines:
            _lines = np.array(list(filter(None, _lines))).flatten()
            HARAnalyzer.__setup_hide_lines(axis.get_figure(), _lines, _legend_lines)

    def plot_buffer_size_graph(self, title, axis=None, references=None, accessory_legend=True):
        """
        Plot the graph representing the amount of buffered media data (in seconds).
        :param title: the title associated to the plotted graph.
        :param axis: the axis where the graph is plot to (if None, a new one is created).
        :param references: a list of "buffer sizes" where a reference line is plotted (in seconds).
        :param accessory_legend: whether the accessory lines are shown in the legend.
        """

        if axis is None:
            _, axis = plt.subplots(1, 1, num=title)

        axis.set_title(title)

        axis.plot(self.buffered_data / 1000, label="Buffer size")

        for _i, _timestamp in enumerate(self.events_info["buffer-flushed"]):
            axis.axvline(_timestamp / 1000, color="C1", linewidth=1, linestyle="--",
                         label="Buffer flushed" if _i == 0 else None)

        if references is not None:
            for _i, _line in enumerate(references):
                _color = "C{}".format((_i + 2) % 4)
                axis.axhline(_line, color=_color, linestyle=":", label="Reference: {:.1f}s".format(_line))

        self._plot_download_timescale_accessory_lines(axis, accessory_legend)

        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Buffer size (seconds)")
        axis.legend()

    def plot_requests_on_off_graph(self, title, axis=None, accessory_legend=True):
        """
        Plot the graph representing the instants when a download request is in progress and the ones it is not.
        :param title: the title associated to the plotted graph.
        :param axis: the axis where the graph is plot to (if None, a new one is created).
        :param accessory_legend: whether the accessory lines are shown in the legend.
        """

        def _intervals_to_lines(_intervals, _y):
            return [[(_lower / 1000, _y), (_higher / 1000, _y)] for _lower, _higher in _intervals]

        if axis is None:
            _, axis = plt.subplots(1, 1, num=title)

        axis.set_title(title)

        _on_intervals = list(IntervalsHelper.merge_intervals(
            map(lambda _chunk: (_chunk["relative-start-time"], _chunk["relative-start-time"] + _chunk["total-time"]),
                self.downloaded_chunks)))
        _off_intervals = IntervalsHelper.opposite_intervals(_on_intervals, 0, self.total_download_time_ms)

        _off, _on = 0.5, 1.5
        _off_intervals_lc = LineCollection(_intervals_to_lines(_off_intervals, _off), colors="C0", linewidth=4)
        _on_intervals_lc = LineCollection(_intervals_to_lines(_on_intervals, _on), colors="C1", linewidth=4)

        axis.add_collection(_off_intervals_lc)
        axis.add_collection(_on_intervals_lc)

        self._plot_download_timescale_accessory_lines(axis, accessory_legend)

        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("On/Off")
        axis.set_ylim(_off - 0.5, _on + 0.5)
        axis.set_yticks([_off, _on])
        axis.set_yticklabels(["Off", "On"])

        if accessory_legend:
            axis.legend()

    def plot_playback_timescale_graph(self, title=None, figsize=None):
        """
        Plots the graph in video playback timescale representing the actual and nominal downloaded bitrates.
        :param title: the title associated to the graph (if none, it is used the HAR filename, without extension).
        :param figsize: the size of the created figure.
        """

        if title is None:
            title = "{} (Playback Timescale)".format(self.filename)

        _f, _axis = plt.subplots(1, 1, sharex="all", num=title, figsize=figsize)
        _f.suptitle(title)

        # Actual bitrate
        _axis.plot(np.trim_zeros(self.playback_bitrates["actual"], trim="b")[:-1] / 1e6, label="Actual Bitrate")

        # Nominal bitrate
        _nominal_bitrate = [[(_lower / 1000, _bitrate / 1e6), (_higher / 1000, _bitrate / 1e6)]
                            for _bitrate, _intervals in self.playback_bitrates["nominal"].items()
                            for _lower, _higher in _intervals]
        _nominal_bitrate_lc = LineCollection(_nominal_bitrate, colors="C1", label="Nominal Bitrate")
        _axis.add_collection(_nominal_bitrate_lc)
        _axis.autoscale_view(True, True, True)

        for _i, _timestamp in enumerate(self.events_info["playback-started-video"]):
            _axis.axvline(_timestamp / 1000, color="C6", linewidth=1, linestyle="--", zorder=-1,
                          label="Playback started" if _i == 0 else None)
        for _i, _timestamp in enumerate(self.events_info["playback-stopped-video"]):
            _axis.axvline(_timestamp / 1000, color="C7", linewidth=1, linestyle="--", zorder=-1,
                          label="Playback stopped" if _i == 0 else None)

        # Reorder the legend entries (otherwise the LineCollection is the last)
        _handles, _labels = _axis.get_legend_handles_labels()
        _handles.insert(1, _handles.pop(-1))
        _labels.insert(1, _labels.pop(-1))
        _axis.legend(_handles, _labels)

        _axis.set_xlabel("Video Time (seconds)")
        _axis.set_ylabel("Bitrate (Mbps)")

    def _plot_playback_bitrate(self, axis, **kwargs):
        """
        Plot the graph representing the variations in the playback bitrates.
        :param axis: the axis where the graph is plot to.
        :param kwargs: additional parameters forwarded to the plot method (e.g. line color, width, label, ...).
        :return: the object representing the plotted line.
        """

        _get_value = np.vectorize(lambda _array, _key: _array[_key])
        _switches = self.events_info["bitrate-switches"]
        if not _switches:
            return None

        _times = np.append(_get_value(_switches, "relative-timestamp"), self.total_download_time_ms)
        _rates = np.append(_get_value(_switches, "bitrate-to"), _get_value(_switches, "bitrate-to")[-1])
        return axis.step(_times / 1000, _rates / 1e6, where="post", **kwargs)

    def _plot_requested_bitrate(self, axis, **kwargs):
        """
        Plot the graph representing the variations in the requested bitrates.
        :param axis: the axis where the graph is plot to.
        :param kwargs: additional parameters forwarded to the plot method (e.g. line color, width, label, ...).
        :return: the object representing the plotted line.
        """

        _segments = []
        for _chunk in self.downloaded_chunks:
            _start_time_ms = _chunk["relative-start-time"]
            _end_time_ms = _start_time_ms + _chunk["total-time"]
            _bitrate = self.mpd_parser.get_nominal_bitrate(_chunk["quality"])

            _segments.append([(_start_time_ms / 1000, _bitrate / 1e6), (_end_time_ms / 1000, _bitrate / 1e6)])

        if "color" in kwargs:
            kwargs["colors"] = kwargs["color"]

        _line_collection = LineCollection(_segments, **kwargs)
        axis.add_collection(_line_collection)
        axis.autoscale_view(True, True, True)
        return [_line_collection, ]

    def _plot_download_timescale_accessory_lines(self, axis, accessory_legend):
        """
        Plot some accessory lines associated to download timescale graphs (i.e. playback start, stop and cdn changes).
        :param axis: the axis where the graph is plot to.
        :param accessory_legend: whether the accessory lines are shown in the legend.
        """

        for _i, _timestamp in enumerate(self.events_info["playback-started"]):
            axis.axvline(_timestamp / 1000, color="C6", linewidth=1, linestyle="--", zorder=-1,
                         label="Playback start" if accessory_legend and _i == 0 else None)
        for _i, _timestamp in enumerate(self.events_info["playback-stopped"]):
            axis.axvline(_timestamp / 1000, color="C7", linewidth=1, linestyle="--", zorder=-1,
                         label="Playback stop" if accessory_legend and _i == 0 else None)
        for _i, _timestamp in enumerate(self.events_info["cdn-changes"]):
            axis.axvline(_timestamp / 1000, color="C8", linewidth=1, linestyle="--", zorder=-1,
                         label="CDN change" if accessory_legend and _i == 0 else None)

    @staticmethod
    def __load_har_file(path: str, filename: str):
        """
        Loads the content of the HAR file.
        :param path: the path where the HAR file is stored.
        :param filename: the name of the HAR file.
        :return: the in-memory representation of the HAR file.
        """

        _har_file_path = os.path.join(path, filename)
        with open(_har_file_path, "r") as _file:
            return json.load(_file)

    @staticmethod
    def __extract_downloaded_chunks_information(har_representation, video_only=True):
        """
        Extracts the main information about the downloaded chunks of data as contained in the HAR file.
        :param har_representation: the in-memory representation of the HAR file.
        :param video_only: whether the chunk types other than video are skipped.
        :return: a numpy arrays containing the information about the downloaded chunks (sorted by start-time).
        """

        _downloaded_chunks = list()
        for _entry in har_representation["log"]["entries"]:

            # Extract the main information from request and response objects
            _request, _response = _entry["request"], _entry["response"]
            _url, _domain, _ext, _method = HARAnalyzer.__extract_request_information(_request)
            _response_code = int(_entry["response"]["status"])

            # Consider only the elements of interest
            if (_ext, _method, _response_code) == (".mp4", "GET", 206):

                # URL format: https://domain/*/*/*_media_quality.ext",
                _media_type = _url.split("_")[-2].upper()
                _media_quality = int(_url.split("_")[-1].split(".")[0])

                if video_only and not _media_type == "VIDEO":
                    continue

                _start_time = dateutil.parser.parse(_entry["startedDateTime"]).timestamp() * 1000

                try:
                    _byte_range = tuple(map(int, HARAnalyzer.__lookup_header(
                        _request["headers"], "Range").split("=")[1].split("-")))
                    _content_length = int(HARAnalyzer.__lookup_header(_response["headers"], "Content-Length"))
                except KeyError:
                    sys.stderr.write(
                        "Error: failed reading chunk information (URL: {}, start-time: {})\n".format(_url, _start_time))
                    continue

                if _byte_range[1] - _byte_range[0] + 1 != _content_length:
                    raise APVanalyzerException("Incompatible content-length and byte-range for URL '{}'".format(_url))

                _downloaded_chunks.append({
                    "url": _url,
                    "domain": _domain,
                    "media-type": _media_type,
                    "quality": _media_quality,

                    "start-time": _start_time,
                    "total-time": _entry["time"],
                    "timings": _entry["timings"],

                    "content-length": _content_length,
                    "byte-range": _byte_range,
                })

        return np.array(sorted(_downloaded_chunks, key=lambda _chunk: _chunk["start-time"]))

    @staticmethod
    def __compute_relative_times_information(downloaded_chunks):
        """
        Given the array containing the information about downloaded media chunks (assumed to be sorted by start-time),
        compute the initial time instant and the relative start times for each chunk.
        :param downloaded_chunks: the array containing the downloaded chunks.
        :return: the initial time instant (in milliseconds).
        """

        _initial_time_ms = downloaded_chunks[0]["start-time"]
        for _chunk in downloaded_chunks:
            _chunk["relative-start-time"] = _chunk["start-time"] - _initial_time_ms

        return _initial_time_ms

    @staticmethod
    def __get_total_download_time(downloaded_chunks):
        """
        Given the array containing the information about downloaded media chunks, returns the total time elapsed between
        the beginning of the download of the first chunk and the completion of the download of the last one.
        :param downloaded_chunks: the array containing the downloaded chunks.
        :return: the total elapsed time (in milliseconds).
        """

        return max(map(lambda _chunk: _chunk["relative-start-time"] + _chunk["total-time"], downloaded_chunks))

    @staticmethod
    def __compute_download_throughput(downloaded_chunks, total_download_time_ms: int):
        """
        Computes the per-second download throughput.
        :param downloaded_chunks: the array containing the downloaded chunks.
        :param total_download_time_ms: the total download time (in milliseconds).
        :return: a numpy array containing the computed throughput information.
        """

        _download_throughput = TimeArrayHelper(total_download_time_ms)
        for _chunk in downloaded_chunks:
            _receive_time_ms = _chunk["timings"]["receive"]
            _start_request_ms = _chunk["relative-start-time"]
            _start_receive_ms = _start_request_ms + _chunk["total-time"] - _receive_time_ms
            _end_receive_ms = _start_request_ms + _chunk["total-time"]

            _download_throughput.increment(_start_receive_ms, _end_receive_ms, _chunk["content-length"] * 8)

        return _download_throughput.array

    @staticmethod
    def __compute_playback_bitrates(downloaded_chunks, mpd_parser):
        """
        Computes the nominal and per-second actual bitrate, depending on the downloaded chunks of data.
        :param downloaded_chunks: the array containing the downloaded chunks.
        :param mpd_parser: the object representing an MPD file.
        :return: a dictionary grouping two numpy array containing the computed bitrate information.
        """

        _nominal_bitrates = defaultdict(list)
        _actual_bitrate = TimeArrayHelper(mpd_parser.get_video_duration())
        _get_initial_byte = np.vectorize(lambda _segment: _segment["byte-range"][0])

        for _chunk in downloaded_chunks:
            _, _mpd_chunk_info = mpd_parser.lookup_chunk(_chunk["quality"], _chunk["byte-range"])
            _start_playback_ms = _mpd_chunk_info["begin-playback-ms"]
            _end_playback_ms = _mpd_chunk_info["end-playback-ms"]
            _chunk_bitrate = mpd_parser.get_nominal_bitrate(_chunk["quality"])

            _nominal_bitrates[_chunk_bitrate].append((_start_playback_ms, _end_playback_ms))
            _actual_bitrate.increment(_start_playback_ms, _end_playback_ms, _chunk["content-length"] * 8)

        return {
            "actual": _actual_bitrate.array,
            "nominal": {_bitrate: np.array(list(IntervalsHelper.merge_intervals(_intervals)))
                        for _bitrate, _intervals in _nominal_bitrates.items()}
        }

    @staticmethod
    def __compute_buffered_data(downloaded_chunks, event_info, mpd_parser, total_download_time_ms: int):
        """
        Computes the amount of buffered media for each download second.
        :param downloaded_chunks: the array containing the downloaded chunks.
        :param event_info: the data structure containing the main events occurred.
        :param mpd_parser: the object representing an MPD file.
        :param total_download_time_ms: the total download time (in milliseconds).
        :return: a numpy array containing, for each element, the amount of buffered data.
        """

        _buffer = BufferHelper(total_download_time_ms, event_info["buffer-flushed"])
        _get_initial_byte = np.vectorize(lambda _segment: _segment["byte-range"][0])

        # Represents whether a video segment has already been considered in the "buffer" size computation
        # --> two segments with different qualities but representing the same video chunk are considered only once
        _segment_considered = [False] * mpd_parser.get_video_chunks_number()

        # Computation of the amount of new buffered data added for each second
        for _chunk in downloaded_chunks:

            _mpd_chunk_idx, _mpd_chunk_info = mpd_parser.lookup_chunk(_chunk["quality"], _chunk["byte-range"])

            # Prevent considering twice the same video chunk
            if not _segment_considered[_mpd_chunk_idx]:
                _download_completed = _chunk["relative-start-time"] + _chunk["total-time"]
                _buffer.data_received(_download_completed, _mpd_chunk_info["duration"])
                _segment_considered[_mpd_chunk_idx] = True

        # Computation of the amount of buffered data consumed for each second
        for _start, _stop in zip(event_info["playback-started"], event_info["playback-stopped"]):
            _buffer.video_playback(_start, _stop)

        return _buffer.buffer

    @staticmethod
    def __extract_mpd_urls(har_representation):
        """
        Extracts and returns the list of urls referencing MPD files.
        :param har_representation:  the in-memory representation of the HAR file.
        :return: the list of MPD urls.
        """

        _mpd_urls = list()
        for _entry in har_representation["log"]["entries"]:
            _url, _, _ext, _method = HARAnalyzer.__extract_request_information(_entry["request"])
            if _ext == ".mpd" and _method == "GET":
                _mpd_urls.append(_url)
        return _mpd_urls

    @staticmethod
    def __extract_events(har_representation, initial_download_time_ms: int, total_download_time_ms: int):
        """
        Extracts the main information from the logs captured in the HAR file.
        :param har_representation:  the in-memory representation of the HAR file.
        :param initial_download_time_ms: the initial download time instant (in milliseconds).
        :param total_download_time_ms: the total download time (in milliseconds).
        :return: a dictionary containing the event logs, the bitrate switches, the instants when the playback started
        and stopped (both in download timescale and playback timescale), the instants when the buffer was flushed.
        """

        _event_log, _bitrate_switches = list(), list()
        _playback_started, _playback_stopped = list(), list()
        _buffer_flushed, _seek_detected = list(), list()

        for _entry in har_representation["log"]["entries"]:

            _request = _entry["request"]
            _url, _, _, _method = HARAnalyzer.__extract_request_information(_request)
            if _url != HARAnalyzer.__player_info_url() or _method != "POST":
                continue

            try:
                for _line in _request["postData"]["text"].splitlines():

                    # Keep only the events of interest
                    if any(_str in _line for _str in HARAnalyzer.__event_filters_accept()):

                        _timestamp = dateutil.parser.parse(_line[57:81]).timestamp() * 1e3
                        _relative_timestamp = _timestamp - initial_download_time_ms
                        _event_log.append((_relative_timestamp, _line[90:]))

                        # Bitrate switch
                        if "Switched bitrate from" in _line:
                            _bitrate_from = int(_line[90:].split(" ")[5].replace("kbps", "")) * 1e3
                            _bitrate_to = int(_line[90:].split(" ")[7].replace("kbps.", "")) * 1e3
                            _bitrate_switches.append({
                                "timestamp": _timestamp,
                                "relative-timestamp": _relative_timestamp,
                                "bitrate-from": _bitrate_from,
                                "bitrate-to": _bitrate_to,
                            })

                        # Playback started
                        if "MainRenderer transitioned" in _line and "to PLAYING" in _line:
                            _playback_started.append(max(0, _relative_timestamp))

                        # Playback stopped
                        if "MainRenderer transitioned from PLAYING" in _line:
                            _playback_stopped.append(min(_relative_timestamp, total_download_time_ms))

                        # Buffer flushed
                        if "[VIDEO] Flushed buffer" in _line:
                            _buffer_flushed.append(_relative_timestamp)

                        # Seek detected
                        if "Seeking on the VideoElement to" in _line:
                            _seek_to = float(_line.split()[-1]) * 1000
                            _seek_detected.append((_relative_timestamp, _seek_to))

            except KeyError as e:
                _timestamp = dateutil.parser.parse(_entry["startedDateTime"]).timestamp() * 1e3
                sys.stderr.write("Error: failed extracting log entry (timestamp: {})\n".format(_timestamp))

        # Add a playback stopped at the end of the capture if not present
        if len(_playback_started) > len(_playback_stopped):
            _playback_stopped.append(total_download_time_ms)

        # Compute the playback start and stop instants in playback timescale
        _playback_started_video, _playback_stopped_video = list(), list()
        _video_position = 0
        for _started, _stopped in zip(_playback_started, _playback_stopped):

            while _seek_detected:
                _timestamp, _seek_to = _seek_detected[0]
                if _timestamp > _started:
                    break

                _video_position = _seek_to
                _seek_detected.pop()

            _playback_started_video.append(_video_position)
            _video_position += _stopped - _started
            _playback_stopped_video.append(_video_position)

        return {
            "playback-started": np.array(_playback_started),
            "playback-stopped": np.array(_playback_stopped),
            "playback-started-video": np.array(_playback_started_video),
            "playback-stopped-video": np.array(_playback_stopped_video),
            "buffer-flushed": np.array(_buffer_flushed),
            "bitrate-switches": _bitrate_switches,
            "event-log": _event_log,
        }

    @staticmethod
    def __extract_cdn_changes(downloaded_chunks):
        """
        Given the array containing the information about downloaded media chunks, returns an array containing the time
        instants when the a new CDN has been selected to download the media chunks.
        :param downloaded_chunks: the array containing the downloaded chunks.
        :return: a numpy array containing the time instants associated to CDN changes (in milliseconds).
        """

        _get_domain = np.vectorize(lambda _chunk: _chunk["domain"])
        _get_relative_start_time = np.vectorize(lambda _chunk: _chunk["relative-start-time"], otypes=[int, ])
        _cdn_changes = _get_domain(downloaded_chunks[1:]) != _get_domain(downloaded_chunks[:-1])
        return _get_relative_start_time(downloaded_chunks[1:][_cdn_changes])

    @staticmethod
    def __extract_request_information(request_object):
        """
        Given an object representing an HTTP request, extracts the url, the web domain, the extension of the referenced
        file, and the request method.
        :param request_object: the object containing the information about the HTTP request.
        :return: a tuple (url, domain, extension, request method).
        """

        _url = request_object["url"]
        _parsed_url = urlparse(_url)
        _, _ext = os.path.splitext(_parsed_url.path)
        return _url, _parsed_url.netloc, _ext, request_object["method"].upper()

    @staticmethod
    def __lookup_header(headers, header_name: str):
        """
        Lookups the header of interest from a set of HTTP headers.
        :param headers: the set of HTTP headers to be considered.
        :param header_name: the name of the header of interest.
        :return: the value associated to the header of interest.
        """

        for _header in headers:
            if _header["name"] == header_name:
                return _header["value"]

        raise KeyError("Header entry for name '{}' not found".format(header_name))

    @staticmethod
    def __player_info_url():
        """
        Returns the URL used to recognize the packets containing event logs.
        :return: the URL used to recognize the packets containing event logs.
        """

        return "https://fls-eu.amazon.com/1/aiv-web-player/1/OE"

    @staticmethod
    def __event_filters_accept():
        """
        Returns the set of identifiers used to select the event logs of interest.
        :return: the set of identifiers used to select the event logs of interest.
        """

        return ["MainRendererStateMachine", "FragmentLoader", "SmoothStreamingSessionState",
                "BasicVideoElementWrapper", "StreamingSessionState",
                "ResolutionBasedVideoQualityFilter", "PixelCountVideoQualityFilter"]

    @staticmethod
    def __setup_hide_lines(figure, lines, legend_lines):

        _lined = dict()
        for line, _legend_line in zip(lines, legend_lines):
            _legend_line.set_picker(5)  # 5 pts tolerance
            _lined[_legend_line] = line

        def _on_line_pick(event):
            _fn_legend_line = event.artist
            _fn_line = _lined[_fn_legend_line]

            _fn_line.set_visible(not _fn_line.get_visible())
            _visible = _fn_line.get_visible()

            # Change the alpha on the line in the legend so we can see what lines have been toggled
            _fn_legend_line.set_alpha(1.0 if _visible else 0.5)
            plt.draw()

        figure.canvas.mpl_connect('pick_event', _on_line_pick)


if __name__ == "__main__":

    # Parse the command line arguments
    _parser = argparse.ArgumentParser()
    _parser.add_argument("har_file", help="The HAR file containing the capture to be analysed")
    _parser.add_argument("--mpd-file", help="The MPD file containing the information about the video representations"
                                            " (automatically downloaded if not specified)")
    _parser.add_argument("--shaper-file", help="The csv file containing the information about shaper configuration")
    _args = _parser.parse_args()

    # Parse the MPD file (if available)
    _mpd_parser = None
    if _args.mpd_file is not None:
        _mpd_path, _mpd_file = os.path.split(_args.mpd_file)
        _mpd_parser = MPDParser(_mpd_path, _mpd_file)

    # Parse the Shaper CSV log file (if available)
    _shaper_parser = None
    if _args.shaper_file is not None:
        _shaper_path, _shaper_file = os.path.split(_args.shaper_file)
        _shaper_parser = ShaperCSVParser(_shaper_path, _shaper_file)

    # Parse the HAR file
    _har_path, _har_file = os.path.split(_args.har_file)
    _har_analyzer = HARAnalyzer(_har_path, _har_file, _mpd_parser, _shaper_parser)

    # Plot the three graphs showing all the most important information
    _har_analyzer.plot_download_timescale_graph()
    _har_analyzer.plot_playback_timescale_graph()
    _har_analyzer.mpd_parser.plot_representation_bitrate_plots()

    plt.show()

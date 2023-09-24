import bisect
import dill
import datetime
import math
import numpy as np
import torch
import os
from collections import defaultdict


class MemmapHDF5Dict():
    """
    Allows for associative access to a memmap/pytables file using the special getitem function
    """

    def __init__(self, filepath, config, file_format=None):
        self.filepath = filepath
        self.config = config
        self.has_index_dim = (self.config["file"]["daterange"] is not None)
        self.file_format = file_format
        self.open()
        pass

    def __enter__(self):
        self.open()

    def __exit__(self):
        self.close()

    def open(self):
        if getattr(self, "fo", None) is not None:
            self.fo.close()

        if self.filepath in ["h5"] or self.file_format == "hdf5":
            import pytables
            self._fo = pytables.open_file(self.filepath, mode="r")
            self.fo = self.fo.getNode("/group0", "table0")
            self.file_format = "hdf5"
        else:
            self.fo = np.memmap(self.filepath,
                                dtype='float32',
                                mode='r',
                                shape=self.config["file"]["dims"],
                                offset=0)
            self.file_format = "memmap"

    def close(self):
        if self.fo is not None:
            self.fo.flush()
            if self.file_format in "hdf5":
                self._fo.close()
            else:
                del self.fo

    def __getitem__(self, index):
        # print("GETITEM: ", index)
        indices, categories = index

        # translate category names into ranges
        cat_rgs = {}
        for category in categories:
            splt = category.split(":")
            if len(splt) == 2:
                if splt[1][-3:] == "hPa":
                    pressure2index = {v: k for k, v in self.config["vbls"][splt[0]]["index2pressure"].items()}
                    idx = pressure2index[int(splt[1][:-3])]
                else:
                    idx = int(splt[1])
                offset = self.config["vbls"][splt[0]]["offset"]
                n_levels = 1 if self.config["vbls"][splt[0]]["levels"] is None else len(self.config["vbls"][splt[0]][
                                                                                            "levels"])
                assert n_levels > idx, "invalid level index!"
                cat_rgs[category] = (offset + idx, offset + idx, self.config["vbls"][splt[0]]["type"] == "temp")
            else:
                offset = self.config["vbls"][category]["offset"]
                n_levels = 1 if self.config["vbls"][category]["levels"] is None else len(
                    self.config["vbls"][category]["levels"])
                cat_rgs[category] = (offset, offset + n_levels - 1, self.config["vbls"][splt[0]][
                    "type"] == "temp")  # CHECK: Indices are inclusive right?

        if isinstance(indices, slice):
            # special instance - we can return a view!
            results_dict = {cat_name: self.fo[indices, slice(rg[0], rg[1] + 1, 1)]
            if rg[2] else self.fo[slice(rg[0], rg[1] + 1, 1)] for cat_name, rg in cat_rgs.items()}
        else:
            results_dict = {}
            for cat_name, rg in cat_rgs.items():
                if indices is not None:
                    results_dict[cat_name] = self.fo[np.ix_(indices, np.arange(rg[0], rg[1] + 1))]
                else:
                    if rg[2]:
                        # print("BRANCH TMP NON INDICES:", rg)
                        results_dict[cat_name] = self.fo[slice(None, None), slice(rg[0], rg[1] + 1)]
                        # self.fo[np.arange(self.fo.shape[0]), np.arange(rg[0], rg[1] + 1)]
                    else:
                        results_dict[cat_name] = self.fo[slice(rg[0], rg[1] + 1)]
                        # np.arange(rg[0], rg[1] + 1)

        return results_dict


class DatafileJoin():

    def __init__(self, datapath):
        self.datapaths = datapath if isinstance(datapath, (list, tuple)) else [datapath]

        self.dataset_config = {"variables": {}, "memmap": {}}
        for datapath in self.datapaths:
            dc = dill.load(open(datapath, "rb"))
            datapath = os.path.dirname(datapath)
            for k, v in dc["variables"].items():
                dc["variables"][k]["mmap_name"] = os.path.join(datapath, dc["variables"][k]["mmap_name"])
            kys = list(dc["memmap"].keys())
            for k in kys:
                dc["memmap"][os.path.join(datapath, k)] = dc["memmap"][k]
                del dc["memmap"][k]
            self.dataset_config["variables"].update(dc["variables"])
            self.dataset_config["memmap"].update(dc["memmap"])

        # Load datasets
        self.datasets = {}
        for filepath, config in self.dataset_config["memmap"].items():
            file_config = {"file": self.dataset_config["memmap"][filepath],
                           "vbls": {k: v for k, v in self.dataset_config["variables"].items() if
                                    v["mmap_name"] == filepath}}
            self.datasets[filepath] = MemmapHDF5Dict(filepath=filepath,
                                                     config=file_config)  # Need switch for H5PY if necessary
        pass

    def _get_file_indices(self, ts_indices, cat_name, config=None):

        cat_name_base = cat_name.split(":")[0]
        if cat_name_base not in self.dataset_config["variables"]:
            print("Cannot retrieve variable {} as it cannot be found in any of the dataset(s) at {}!".format(cat_name,
                                                                                                             self.datapaths))
            return None, False

        file_filename = self.dataset_config["variables"][cat_name_base]["mmap_name"]

        file_coords = None
        if ts_indices is not None:
            if isinstance(ts_indices, tuple):
                ts_min = ts_indices[0]
                ts_max = ts_indices[1]
                tfreq = ts_indices[2] if len(ts_indices) > 2 else None
            else:
                ts_min = min(ts_indices)
                ts_max = max(ts_indices)
                tfreq = None
                ts_ix = ts_indices

            if self.dataset_config["memmap"][file_filename]["daterange"] is not None:
                if ts_min < self.dataset_config["memmap"][file_filename]["daterange"][0] \
                        or ts_max > self.dataset_config["memmap"][file_filename]["daterange"][1]:
                    return None, False

                # deal with fractional strides and coordinate mismatches
                tfreq_s_ds = self.dataset_config["memmap"][file_filename]["tfreq_s"]
                tfreq_s = tfreq_s_ds if tfreq is None else tfreq
                is_tuple = isinstance(ts_indices, tuple)
                if is_tuple:
                    if not (tfreq_s//tfreq_s_ds and not tfreq_s%tfreq_s_ds):
                        ts_ix = np.linspace(ts_indices[0], ts_indices[1],
                                            int((ts_indices[1] - ts_indices[0]) // tfreq_s))
                        file_coords = (ts_ix-self.dataset_config["memmap"][file_filename]["daterange"][0]) / tfreq_s_ds
                        # will be postprocessed by the second if branch below!
                    else:
                        ts_start = (ts_indices[0] -
                                    self.dataset_config["memmap"][file_filename]["daterange"][0]) // tfreq_s
                        ts_stop = (ts_indices[1] -
                                   self.dataset_config["memmap"][file_filename]["daterange"][0]) // tfreq_s
                        ts_step = tfreq_s // tfreq_s_ds
                        file_coords = slice(int(ts_start), int(ts_stop), int(ts_step))
                else:
                    file_coords = (np.array(ts_indices) - self.dataset_config["memmap"][file_filename]["daterange"][
                        0]) // tfreq_s_ds

                if not isinstance(file_coords, slice):
                    if config.get("interpolate", "NaN") == "nearest_past":
                        file_coords = np.floor(file_coords).astype(np.int)
                    elif config.get("interpolate", "NaN") == "nearest_future":
                        file_coords = np.ceil(file_coords).astype(np.int)
                    else:
                        file_coords = np.ma.array(file_coords,
                                                  mask=(file_coords - np.floor(file_coords)) != 0.0,
                                                  dtype=int,
                                                  fill_value=
                                                  self.dataset_config["memmap"][file_filename]["daterange"][
                                                      0])

                        # assign_dict_indices[file_filename] = file_coords
        else:
            # assign_dict_indices[file_filename] = None
            file_coords = None

        return file_coords, True

    def __getitem__(self, index):
        """
        Middleware between multiple datafiles (h5py or Memmap, doesn't matter) and Dataset object.

        :param index:
        :return:
        """

        if isinstance(index, str):
            index = (None, [index], None)

        ts_indices = index[0]
        cat_names = index[1]
        config = index[2]

        # Assign categories to be extracted from the correct files, and verify variables are available for the
        # given timeranges
        assign_dict = defaultdict(lambda: [])
        assign_dict_indices = {}
        unavailable_cats = []
        for cat_name in cat_names:
            cat_name_base = cat_name.split(":")[0]
            if cat_name_base not in self.dataset_config["variables"]:
                print(
                    "Cannot retrieve variable {} as it cannot be found in any of the dataset(s) at {}!".format(cat_name,
                                                                                                               self.datapaths))
                unavailable_cats.append(cat_name)
                continue

            file_filename = self.dataset_config["variables"][cat_name_base]["mmap_name"]
            if file_filename not in assign_dict_indices:
                file_coords, status = self._get_file_indices(ts_indices, cat_name, config)
                if not status:
                    unavailable_cats.append(cat_name)
                    continue
                else:
                    assign_dict_indices[file_filename] = file_coords
                assign_dict[file_filename].append(cat_name)

        results_dict = {}
        for k, v in assign_dict.items():
            indices = assign_dict_indices.get(k, None)
            res = self.datasets[k][(indices, v)]
            results_dict.update(res)

        if len(unavailable_cats):
            print("UNAVAILABLE CATS: ", unavailable_cats)
        if len(results_dict.keys()) > 1:
            return results_dict
        elif not results_dict:
            return None
        else:
            return results_dict[list(results_dict.keys())[0]]


class RainbenchDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, datapath,
                 partition_conf,
                 partition_type,
                 sample_conf,
                 partition_selected="train"):

        # Initialization
        self.sample_conf = sample_conf
        self.datapath = datapath
        self.partition_conf = partition_conf
        self.partition_type = partition_type
        self.partition_selected = partition_selected
        self.debug_mode = False  # return sample stride indices instead of sample
        self.get_ts_only_mode = False  # in ts_only_mode, __getitem__ only returns sample timestamps instead of values

        self.sample_mode_order = sorted(self.sample_conf.keys(), key=lambda x: int(x.split("_")[-1]))
        # NOTE: may have to adjust

        # Load Dataset
        self.dataset = DatafileJoin(self.datapath)

        # Calculate sample spreads
        self._sample_spreads = {}
        self._sample_offsets_left = {}
        for sample_mode_name, sample_mode in self.sample_conf.items():
            lowest_t = 0
            highest_t = 0
            for sample_section_name, sample_section in sample_mode.items():
                for vbl_section_name, vbl_section in sample_section.items():
                    if "t" in vbl_section:
                        lowest_t = min(lowest_t, min(vbl_section["t"]))
                        highest_t = max(highest_t, max(vbl_section["t"]))
            self._sample_spreads[sample_mode_name] = highest_t - lowest_t
            self._sample_offsets_left[
                sample_mode_name] = - lowest_t  # NOTE: doesn't work if lowest_t > 0 *UNREALISTIC*

        # Select partition
        self.select_partition(self.partition_selected)

    def get_file_indices_from_ts_range(self, ts_range, vbl_type, tfreq=None, expand=True):
        rg = self.dataset._get_file_indices((*ts_range, tfreq), vbl_type, {})
        if expand and isinstance(rg[0], slice):
            ret = np.arange(rg[0].start, rg[0].stop, (rg[0].step if rg[0].step is not None else 1))
        else:
            ret = rg[0]
        return ret
        #return rg[0].data[0], rg[0].data[-1]

    def get_partition_ts_segments(self, partition_selected):
        """
        Returns an array of all contiguous timestamp ranges within a given partition
        """
        if self.partition_type in ["range"]:
            print("return range indices..")
            return [self.partition_conf[partition_selected]["timerange"]]

        elif self.partition_type in ["repeat"]:
            print("return repeat indices...")
            segs = []
            selected_partition_id = [v["name"] for v in self.partition_conf["partitions"]].index(partition_selected)
            partition_element_offset = (0 if self.selected_partition_id == 0 else sum(
                v["len_s"] for v in self.partition_conf["partitions"][:selected_partition_id]))
            cur_ts = self.partition_conf["timerange"][0] + partition_element_offset
            tot_repeat_partition_element = sum(v["len_s"] for v in self.partition_conf["partitions"])
            while cur_ts < self.partition_conf["timerange"][1]:
                upper_ts = min(cur_ts + self.partition_conf["partitions"][selected_partition_id]["len_s"],
                               self.partition_conf["timerange"][1])
                segs.append((cur_ts, upper_ts))
                cur_ts += tot_repeat_partition_element
            return segs
        else:
            raise Exception()

        return sorted(list(idx_set))

    def select_partition(self, partition_selected):
        self.partition_selected = partition_selected
        # Calculate sample type lengths, and also total lengths for different modes
        self.n_all_samples = 0
        self.n_samples = {}
        if self.partition_type == "repeat":
            self.selected_partition_id = [v["name"] for v in self.partition_conf["partitions"]].index(
                self.partition_selected)

            self.timerange_ts = self.partition_conf["timerange"]
            # Calculate total repeat partition_element
            tot_repeat_partition_element = sum(v["len_s"] for v in self.partition_conf["partitions"])
            # For each sample type, determine how many samples there are in the given partitioning
            for sample_mode_name, sample_mode in self.sample_conf.items():
                sample_spread_s = self._sample_spreads[sample_mode_name]

                # Tot repeat partition_element = [sec1, sec2, sec3] which are repeated then
                # How often does the tot_repeat_partition_element fit into the whole dataset range?
                timerange_ts = self.timerange_ts

                # NOTE: We chop off data belonging to incomplete partition_elements at end of dataset and don't use it
                n_partition_elements = int((timerange_ts[1] - timerange_ts[0]) // tot_repeat_partition_element)

                increment_s = self.partition_conf["partitions"][self.selected_partition_id]["increment_s"]
                len_s = self.partition_conf["partitions"][self.selected_partition_id]["len_s"]
                n_samples_per_partition_element = math.floor(
                    len_s - sample_spread_s) // increment_s + 1  # NOTE: CHECK THIS!

                n_tot_samples = n_samples_per_partition_element * n_partition_elements
                self.n_samples[sample_mode_name] = n_tot_samples
                self.n_all_samples += n_tot_samples
                self.selected_partition_increment_s = increment_s

        elif self.partition_type == "range":

            for sample_mode_name, sample_mode in self.sample_conf.items():
                sample_spread_s = self._sample_spreads[sample_mode_name]
                increment_s = self.partition_conf[self.partition_selected]["increment_s"]

                len_s = self.partition_conf[self.partition_selected]["timerange"][1] - \
                        self.partition_conf[self.partition_selected]["timerange"][0]
                n_samples_per_partition_element = math.floor(len_s - sample_spread_s) // increment_s + 1
                n_tot_samples = n_samples_per_partition_element

                self.n_samples[sample_mode_name] = n_tot_samples
                self.n_all_samples += n_tot_samples
                self.selected_partition_increment_s = increment_s

        else:
            raise NotImplementedError()

        self.sample_mode_binning = np.array([-1] + [self.n_samples[st] for st in self.sample_mode_order]).cumsum()
        return

    def get_sample_at(self, sample_mode_id, sample_ts, sample_idx=None):

        if isinstance(sample_mode_id, str):
            sample_mode_id_tmp = self.sample_mode_order.index(sample_mode_id)
            assert sample_mode_id_tmp != -1, "Unknown sample mode id: {}".format(sample_mode_id)
            sample_mode_id = sample_mode_id_tmp

        sample_conf = self.sample_conf[self.sample_mode_order[sample_mode_id]]
        sample_results = {}
        indices_sampled = []
        ts_sampled = []
        for sample_section_name, sample_section in sample_conf.items():
            sample_results[sample_section_name] = {}
            for vbl_section_name, vbl_section in sample_section.items():
                if vbl_section_name[:len("__const__")] == "__const__":
                    sample_results[sample_section_name][vbl_section_name] = vbl_section["val"]
                    continue
                vbl_name = vbl_section["vbl"]
                if vbl_name in ["__dummy__"]:  # dummy variables can be used to enforce variable spreads
                    continue

                if "t" not in vbl_section:
                    if not self.get_ts_only_mode:
                        sample_results[sample_section_name][vbl_section_name] = np.copy(self.dataset[vbl_name])
                else:
                    vbl_t = vbl_section["t"](sample_ts, sample_idx) if callable(vbl_section["t"]) else vbl_section["t"]
                    if not self.get_ts_only_mode:
                        sample_results[sample_section_name][vbl_section_name] = \
                            self.dataset[(vbl_t + sample_ts,
                                          [vbl_name],
                                          {"interpolate": vbl_section.get("interpolate", "NaN")})]
                    sample_results[sample_section_name][vbl_section_name + "__ts"] = vbl_section["t"] + sample_ts
                    if self.debug_mode:
                        ts_sampled.append(vbl_t + sample_ts)
                        indices_sampled.append(self._get_file_indices(vbl_t + sample_ts, vbl_name, None))
                    if not isinstance(sample_results[sample_section_name][vbl_section_name], np.ndarray):
                        print(vbl_section_name, type(sample_results[sample_section_name][vbl_section_name]))

                # "aggregation mode" allows for sample section slices to be aggregated over time in various ways
                agg_mode = vbl_section["agg_mode"] if "agg_mode" in vbl_section else None
                if agg_mode is not None:
                    if callable(agg_mode):
                        sample_results[sample_section_name][vbl_section_name] = \
                            agg_mode(sample_results[sample_section_name][vbl_section_name])
                    elif agg_mode in ["sum"]:
                        sample_results[sample_section_name][vbl_section_name] = \
                            np.sum(sample_results[sample_section_name][vbl_section_name], axis=0, keepdims=True)
                    elif agg_mode in ["mean"]:
                        sample_results[sample_section_name][vbl_section_name] =\
                            np.mean(sample_results[sample_section_name][vbl_section_name], axis=0, keepdims=True)
                    elif agg_mode in ["max"]:
                        sample_results[sample_section_name][vbl_section_name] = \
                            np.max(sample_results[sample_section_name][vbl_section_name], axis=0, keepdims=True)
                    elif agg_mode in ["min"]:
                        sample_results[sample_section_name][vbl_section_name] = \
                            np.min(sample_results[sample_section_name][vbl_section_name], axis=0, keepdims=True)

        if self.debug_mode:
            return ts_sampled, indices_sampled

        return sample_results

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_all_samples

    def __getitem__(self, index):

        if isinstance(index, (tuple, str)):
            """
            Dataset Access mode 1:

            Retrieve one or several data categories given by cat_indices [lst of strings, or None if all]
            at timestamp indices ts_indices [lst of either timestamps or tuples (start_ts, stop_ts, step_ts)]
            using configuration parameters given by config_dict, e.g.
            config_dict = {"interpolation": []}
            or None if no configuration required 
            Index: (cat_indices, ts_indices, config_dict)

            Returns dictionary of values or views (depending on contiguity of the data accessed)
            """
            return self.dataset[index]

        if not isinstance(index, list):
            """
            Dataset Access mode 2:

            Retrieve one or several samples from the dataset given the sample_configuration. 
            index needs to be a single index or a list of indices to be retrieved.
            Returns dictionary of values or views (depending on contiguity of the data accessed)
            """
            index = [index]

        # Select sample
        assert max(index) < self.n_all_samples, "index {} out of range for dataset length {}".format(index,
                                                                                                     self.n_all_samples)

        # identify which sample type the index corresponds to MODE NOT TYPE
        sample_mode_ids = [bisect.bisect_left(self.sample_mode_binning, i) - 1 for i in index]

        # calculate the central timestamps of the sample(s) to be retrieved
        if self.partition_type == "repeat":

            # Calculate the whole temporal length of repeating unit of the partitioning
            tot_repeat_partition_element = sum(v["len_s"] for v in self.partition_conf["partitions"])

            # Obtain increment and length of the currently selected partition
            increment_s = self.partition_conf["partitions"][self.selected_partition_id]["increment_s"]
            len_s = self.partition_conf["partitions"][self.selected_partition_id]["len_s"]

            # Calculate the offset of the partition element
            partition_element_offset = (0 if self.selected_partition_id == 0 else sum(
                v["len_s"] for v in self.partition_conf["partitions"][:self.selected_partition_id]))

            # Obtain the sample spreads
            sample_spread_s_lst = [self._sample_spreads[self.sample_mode_order[sample_mode_id]] for sample_mode_id in
                                   sample_mode_ids]

            # Determine the sample modes offset that each given index falls into
            sample_mode_offset_idx_lst = [index - self.sample_mode_binning[sample_mode_id] for sample_mode_id in
                                          sample_mode_ids]

            # Calculate how many samples fit into a single one of the currently selected partition elements
            n_samples_per_partition_element = [math.floor(len_s - sample_spread_s) // increment_s + 1 for
                                               sample_spread_s in sample_spread_s_lst]

            # Calculate the repetition of the currently selected partition element that this sample falls into
            partition_n = [int(sample_mode_idx // n_samples_per_partition_element) for sample_mode_idx in
                           sample_mode_offset_idx_lst]

            # Calculate the offset of the sample within the partition element repetition of interest
            partition_n_offset_lst = [int(off_idx % n_samples_per_partition_element) for off_idx in
                                      sample_mode_offset_idx_lst]

            # Calculate the sample mid-center timestamps
            sample_ts_lst = [self.timerange_ts[0] + \
                             _partition_n * tot_repeat_partition_element + \
                             partition_element_offset + \
                             partition_n_offset * increment_s + \
                             self._sample_offsets_left[self.sample_mode_order[sample_mode_id]] for
                             _partition_n, partition_n_offset, sample_mode_id in
                             zip(partition_n, partition_n_offset_lst, sample_mode_ids)]

        elif self.partition_type == "range":

            # Obtain increment and length of the currently selected partition
            increment_s = self.partition_conf[self.partition_selected]["increment_s"]
            len_s = self.partition_conf[self.partition_selected]["timerange"][1] - \
                    self.partition_conf[self.partition_selected]["timerange"][0]

            # Obtain the sample spreads
            sample_spread_s_lst = [self._sample_spreads[self.sample_mode_order[sample_mode_id]] for sample_mode_id in
                                   sample_mode_ids]

            # Calculate how many samples fit into a single one of the currently selected partition elements
            n_samples_per_partition_element = [math.floor(len_s - sample_spread_s) // increment_s + 1 for
                                               sample_spread_s in sample_spread_s_lst]

            # Determine the sample modes offset that each given index falls into
            sample_mode_offset_idx_lst = [index - self.sample_mode_binning[sample_mode_id] for sample_mode_id in
                                          sample_mode_ids]

            # Calculate the offset of the sample within the partition element repetition of interest
            partition_n_offset_lst = [int(off_idx % n_samples_per_partition_element) for off_idx in
                                      sample_mode_offset_idx_lst]

            sample_ts_lst = [self.partition_conf[self.partition_selected]["timerange"][0] + \
                             partition_n_offset * increment_s + \
                             self._sample_offsets_left[self.sample_mode_order[sample_mode_id]] for
                             partition_n_offset, sample_mode_id in
                             zip(partition_n_offset_lst, sample_mode_ids)]

        else:
            raise NotImplementedError()

        if self.debug_mode:
            # DEBUG mode exists in order to feed back which actual *timestamp ranges* have been covered by a particular
            # sample. That does NOT mean that within this range, all variables have been densely sampled! It just
            # returns a limit interval of what could have been sampled for debugging purposes.
            ts_dct = {self.sample_mode_order[smid]: (int(ts) - self._sample_offsets_left[self.sample_mode_order[smid]],
                                                     int(ts) - self._sample_offsets_left[self.sample_mode_order[smid]] +
                                                     self._sample_spreads[self.sample_mode_order[smid]]) for ts, smid in
                      zip(sample_ts_lst, sample_mode_ids)}
            return ts_dct

        # Now actually load the sample data requested (NOTE: Can be performance-improved through compiling indices)
        # First, we need to assign the different data categories to the different memmap files
        results = []
        indices_sampled = []
        for i, (sample_mode_id, sample_ts) in enumerate(zip(sample_mode_ids, sample_ts_lst)):
            sample_results = self.get_sample_at(sample_mode_id, sample_ts, index)
            sample_results["__sample_modes__"] = self.sample_mode_order[sample_mode_id]
            sample_results["__sample_ts__"] = sample_ts_lst
            results.append(sample_results)

        if self.debug_mode:
            all_indices = set()

        return results


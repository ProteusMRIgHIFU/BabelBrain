import glob
import os

class Dataset:
    def __init__(self,test_data_path,id,trajectories):
        dataset_filepath = test_data_path + id + os.sep
        self.id = id
        self.truth_data_filepath = dataset_filepath + "Truth" + os.sep
        self.SimNIBS_folder = glob.glob(dataset_filepath + 'm2m*')[0] + os.sep
        
        # Grab all trajectories.
        # Currently test 4: Deep Brain, Superficial Brain, Skull, Outside Head
        trajectory_folder = dataset_filepath + 'Trajectories' + os.sep
        self.trajectories = {}
        for file in glob.glob(trajectory_folder + '*.txt'):
            for traj in trajectories:
                if traj in file:
                    self.trajectories[traj] = file

        # Grab all scans
        self.CT_type = {}
        for file in glob.glob(dataset_filepath + '*.nii*'):
            if "T1W" in file:
                self.T1W_file = file
            elif "CT" in file:
                self.CT_file = file
                self.CT_type = 'real CT'
            elif "ZTE" in file:
                self.CT_file = file
                self.CT_type = 'ZTE'
            elif "PETRA" in file:
                self.CT_file = file
                self.CT_type = 'PETRA'
            else:
                print(f"{file} is not used in testing")

    def get_valid_trajectories(self):
        valid_keys = ['Deep_Target', 'Superficial_Target','Skull_Target']
        trajectories_subset = dict(filter(lambda i:i[0] in valid_keys, self.trajectories.items()))
        return trajectories_subset
    
    def get_invalid_trajectories(self):
        valid_keys = ['Outside_Target']
        trajectories_subset = dict(filter(lambda i:i[0] in valid_keys, self.trajectories.items()))
        return trajectories_subset
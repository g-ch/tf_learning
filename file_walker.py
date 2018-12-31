"""
Author: Clarence
Date: 2018.12.25

This is a python file to get all the file names in a folder and its subfolder. Usage:

import file_walker

dir="xxx"
scan=file_walker.ScanFile(dir)
files=scan.scan_files()

"""

import os


def case_insensitive_sort(liststring):
    listtemp = [(x.lower(), x) for x in liststring]
    listtemp.sort()
    return [x[1] for x in listtemp]


class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix=None):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):

        print "Scan started!"
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''
            counter = 0
            list = []
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath, special_file))
                else:
                    counter += 1
                    list.append(os.path.join(dirpath, special_file))
                    # files_list.append(os.path.join(dirpath,special_file))
                    # print counter

            if counter > 2:
                print "Found ", counter, " files"
                files_list.extend(list)

        files_list = case_insensitive_sort(files_list)
        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list
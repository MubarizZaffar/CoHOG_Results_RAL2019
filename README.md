# CoHOG_Results_RAL2019
This repository contains the results for our publication titled "CoHOG: A Light-weight, Compute-efficient and Training-free Visual Place Recognition Technique for Changing Environments".

The Python implementation of our work is also provided. If you find our work useful, please cite our publication titled above using the below bibtex.
```
@article{zaffar2020cohog,
  title={CoHOG: A Light-Weight, Compute-Efficient, and Training-Free Visual Place Recognition Technique for Changing Environments},
  author={Zaffar, Mubariz and Ehsan, Shoaib and Milford, Michael and McDonald-Maier, Klaus},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={1835--1842},
  year={2020},
  publisher={IEEE}
}
```
############################################################################################

Update:

Been recieving some queries from users who are executing the main.py file from terminal and cannot see any output. See below:

If you are running the main.py file from terminal, you need to modify the main file, such that the least indented code in main.py is wrapped in a function, say foo(), and then at the end of the main file, add the following:
```
if __name__=="__main__"
    foo()
```

My uploaded version of the code is for users that are using some IDE (like Spyder) to execute the main file. If you open the main.py file from an IDE like Spyder, you'll be able to execute the code without any changes. Many thanks.
#############################################################################################

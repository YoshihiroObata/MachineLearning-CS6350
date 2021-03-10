This is a folder that contains python scripts to perform decision tree boosting, bagging, and random forests.

The .sh file in this script will first run the implementations of each method and produce plots and text file summaries of the results.

It will then run the bagging and random forest experiments in that order. YOU SHOULD NOT RUN THE EXPERIMENTS IN PARALLEL (for now)
 - This is because the name of the csv file of the recorded hypotheses are tied to the number of csv files in the directory currently
 - This will hopefully be implemented better in the future, but it works for now so keep that in mind before running
 - The experiment scripts will produce csv files of the hypothesis for each of the iterations for every test example for both single trees and the ensemble method

The csv files that are already in the folder are from runs of the experiment with bagged trees and random forests made of 500 trees and 100 iterations
This takes about 6 hours using my code, so I saved the person who runs this some time by setting the iterations and the number of trees lower.
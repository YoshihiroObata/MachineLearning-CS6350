# Basic Decision Trees
This repo contains files for training and applying a decision tree using the ID3 algorithm with entropy, majority error, and gini index methods of information gain on a toy dataset.

## What is a decision tree?
At the core, a decision tree is a hierarchical structure that represents a way to make choices. They are a bit like a flowchart, but only flowing down to the final classifications; certainly you've seen flowcharts in your life in school, work, or memes. Decision trees are not just used to help you make choices in life, they are also efficacious in computer science. In the context of machine learning and data science, a decision tree is a learning method that uses logical operations to classify or perform regression on a dataset. It's a simple algorithm in concept, but pretty nifty when used on the right type of dataset.

## How do decision trees work?
Decision trees aim to create a network of choices that can represent a dataset well using examples. Each "node" on the tree is a data subset. The data subset at a node is represented by a "feature" of the data. At each node, branches are created for each value that node can take. Leaves of the tree are class labels and indicate when to stop subsetting the data.

### How is the best feature selected for a node?
To determine the best feature for a node to split on, a technique called information gain can be used on the attributes of a feature. In concept, information gain performs exactly what it sounds like, it calculates a representation of how much information will be gained if a certain attribute is split upon. Of the common ways to implement information gain, I implemented entropy, Gini index, and majority error. Here's a brief decription of how they work. Information gain is a quite nifty, I probably won't be able to explain it as nicely as other resources online, so here are a few 

## Why use decision trees?

## Decision tree example 

# ai-irs-research

Emmet Houghton, Dr. Michael Lee

Last modified: 6/17/21

## Research Project Overview: AI and Management Science 

## Description:
In recent years, there has been an increasing focus on big data and the potential for modern data science technology to add value to businesses, governments, investors, and individuals by changing the way business analytics are determined and used. Although the volume and complexity of many datasets make it difficult for many organizations to use them effectively, by leveraging the power of artificial intelligence, organizations can create a tool that reveals relevant information and variables buried in the data. Specifically, emerging machine learning techniques can be deployed to help entities understand patterns, uncover vulnerabilities, and predict the likelihood of certain outcomes with greater precision and confidence. 

In this project I aim to apply machine learning to identify key growth factors and early indicators of decline for corporations. I will develop a predictive model using public data released by the IRS to demonstrate how certain variables in the data can be used to help understand the effectiveness and sustainability of a current business or service model. Once determined, these variables could be used by managers or analysts that may not have access to large datasets to evaluate individual businesses. 
In addition, the methods developed in this project could be applied in the future by others looking to extract insights from trained neural networks or analyze intelligent machines at a deeper level. By replicating these methods, other types of organizations or communities will be able to better understand and interpret their data.
Methods:

Using historical IRS data, I will train a dense neural network of a shape not yet determined to fit a model of an organization’s growth. The inputs used to train the network as well as the network’s structure will likely evolve over the course of the project. The fully-trained machine will be able to predict future growth or decline of an organization and can be used to identify key independent variables for these metrics.

## Resources and Requirements:
- The dataset of IRS 990 and 990-EZ filings can be found on the AWS Registry of Open Data (resource name: arn:aws:s3:::irs-990-spreadsheets) and can be downloaded as .csv files. The dataset contains IRS forms filed by non-profit organizations between 2010 and 2017.
- The project is written in Python 3.7.6. 
- The repository containing all code for this project will be uploaded here: https://github.com/Emmet-exe/ai-irs-research. All libraries required to replicate the experiments will be frozen in requirements.txt. 

## Schedule:
##### Stage 1 | Preparation
Organize training data and set up the necessary infrastructure in python for machine learning
##### Stage 2 | Modeling
Fit a dense neural network to the dataset by gradient descent
##### Stage 3 | Analysis and Report
Manipulate the model to identify key variables and formalize research conclusions
##### Stage 4 | Finalize Publication
Draft report of findings


# Project Solution Workflow
<img src= https://github.com/RamiFisher1992/rs-amazon/blob/be4767a7ea1f59e29e67bf3c6230f5751acc0d38/readme-image/app-image.PNG alt="Alt Text" width="500" height="300">

This document outlines the workflow for our project solution, detailing the various stages, methodologies, and tools used to achieve the desired results.

## Table of Contents

- [Project Solution Workflow](#project-solution-workflow)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Stages](#project-stages)
    - [Data Preprocessing](#data-preprocessing)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Recommendation Approaches](#recommendation-approaches)
    - [Application Development](#application-development)
  - [Running the Application](#running-the-application)

## Introduction

This document provides an overview of the workflow employed to deliver my project solution. It covers data preprocessing, exploratory data analysis, recommendation approaches, and the development of the application using Flask.

## Project Stages

### Data Preprocessing

- **data_preproccessing.ipynb:** This initial stage focuses on data preprocessing and cleaning. The output includes two key files:
   - `preprocessed.csv`
   - Usernames and Items IDs pairs in JSON format

### Exploratory Data Analysis (EDA)

- **Eda.ipynb:** In this stage, we perform exploratory data analysis by visualizing insights from the `preprocessed.csv` dataset.

### Recommendation Approaches

- We explore and compare three recommendation approaches, each implemented in separate Jupyter notebooks:
   - **Collaborative Filtering (CF) (includes Random model):**
   - **Matrix Factorization:** 
   - **Content-Based:** 

- Each notebook provides a summary explanation at the top and generates relevant files for recommendation in the app, all saved in the 'outputs' directory.
   - *Note:* CF models will not be used in the app.

### Application Development

- The recommendation application is implemented using Flask, with relevant files stored in the 'src' directory.
- Three models are optional for recommendation within the application:
   - Random
   - SVD
   - Content-Based (cosine or Euclidean)

## Running the Application

To run the application locally, follow these steps:

1. Clone this repository to your local system.
2. Install all the required libraries mentioned in the `requirements.txt` file.
3. Open the terminal from your project directory and execute the command: `python app.py`.
4. Open your web browser and enter the following address in the address bar: [http://127.0.0.1:8080/](http://127.0.0.1:8080/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



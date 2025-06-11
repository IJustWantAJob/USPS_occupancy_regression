# Webscraping/Regression

# USPS Occupancy Regression Tool

## **1. Introduction.**

There are tens of thousands of postal buildings, and it can be challenging to track and know which ones need to be retrofitted first. Many buildings also lack information on this, so I wanted to create a machine learning (regressor) model that can find a postal building’s ‘building occupancy date’ based on other information on the building. This can help find a way to identify the age of a post office building in the scenario where it isn’t known.

The data I want to use comes from USPS and was released under the Freedom of information Act, yet isn’t easily accessible in the form of a dataset. Instead the file for each state is in the form of a .xls/.csv for download. Because of various rules and regulations, it’s tough to find a compiled dataset of this information online as well, outside of the usps.com site.

[](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfs53qTyrrGaqA5z_CCd6mY61Y43gVZl8HrTesUoLLlR1LESf9nBHqOUj6MlwsaWcgCL7Souc47yqO1Y6suN4UTVyGv3rmzwkk3rxOLiVeb0ePPtteQjvH88MkXtYYpbcfquA9-3A?key=R5FlYMn0-LbVM6hE71p9FT1j)

To gather this data manually and merge all the CSV files together would take a lot of time and be extremely boring, so I decided to build a web scraper which would allow me to scrape the USPS site and download each file to my computer, allowing me to run a script and merge the files together.

At first I attempted to directly access the URL with a .get request from my terminal, but it was instantly blocked. I soon realized that USPS blocks such attempts of web scraping, and I’d need to find a way around such blocks.

One method to do so was mimicking a web browser, which a typical user would have when accessing a site, so I create a fake one.

```python
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
```

After receiving a successful status code from my .get request, I used an open source tool called 'BeautifulSoup' which allowed me to fetch HTML content.

> BeautifulSoup parses HTML which makes it easier to navigate, search, and access data on a site's page.
> 

I noticed while looking at the site that all the files had a hyperlink for downloading the .csv files, so I searched specifically for
all <a> tags (which represent hyperlinks). I focused only on tags with the href attribute, and those that ended in '.csv'. I then appended every such instance to a list that contained the complete URL to handle relative links (so all links are absolute).

```python
soup = BeautifulSoup(response.content, 'html.parser')
csv_links = [] # empty, will be filled with complete links to csv

for link in soup.find_all('a', href=True): # This finds all <a>
# tags which represents hyperlink. The href argument ensures
# only tags with href attribute (specified link URL) are included
    href = link['href'] # all href or all links 

    if href.endswith('.csv'):
        full_url = requests.compat.urljoin(url,href)
        # This forms a complete URL to handle relative links so 
        # all links are absolute
        csv_links.append(full_url)
```

Afterwards I made a directory in my computer to store the results, which would allow me to identify any issues that may arise (and not completely ruin my downloads folder with junk).

While attempting to send .get requests to these csv download links, I found that a 'pause' of 2 seconds helps my get requests from being blocked, so I added in a command to slow down our code

```python
    time.sleep(2)
```

I also decided to save the last 2 letters from each csv file, because they contained information on each state (such as ne (Nebraska) or la (Louisiana))
This allowed me to save each file as 'file_{state_prefix}.csv' for easier handling later on. I was able to accomplish this by using the file.write command.

After finally running the script, our terminal output printed out the following:

```
200
```

200 here represents the ‘OK’ status code in an HTTP response (our request was successful)

```
['https://about.usps.com/who/legal/foia/documents/owned-facilities/al.csv', 
'https://about.usps.com/who/legal/foia/documents/owned-facilities/ak.csv', 
'https://about.usps.com/who/legal/foia/documents/owned-facilities/az.csv', 
'https://about.usps.com/who/legal/foia/documents/owned-facilities/ar.csv', 
'https://about.usps.com/who/legal/foia/documents/owned-facilities/ca.csv'
...
```

I added a print statement to ensure that we’re accessing the correct download links, which seems to be the case by manually inspecting a few of the download links.

```
Downloaded webscrape_project_cse40/csv_results/file_al.csv
Downloaded webscrape_project_cse40/csv_results/file_ak.csv
Downloaded webscrape_project_cse40/csv_results/file_az.csv
...
Downloaded webscrape_project_cse40/csv_results/file_wy.csv
All downloaded
```

At this point we have a directory with many .csv files that contain the information that we need. Yet the information is very far apart and difficult to access (in its current format) so I wanted to focus on consolidating it into one dataframe.

After accessing just one csv as a dataframe inside my python notebook I noticed that by dropping 2 rows, it makes aligns the true column names with the top of the dataframe (this was also true for other dataframes).

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image.png)

One file individually looks great! Let’s access every file and append them into one dataframe. Because each dataframe has the same column names, we can keep our number of columns the same, and instead add on more rows (for each state).

```python
csv_files = glob.glob(f"{csv_dir}/*.csv")
```

After appending every csv into one dataframe, we finally have our combined dataframe to finally clean and use for our models.

## 2. Data Cleaning

I decided to look at the dataframe and see what our data looks like.

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%201.png)

Our dataframe has a few new columns, with some looking like they’re state specific (OHIO 1 for example could be an excess column from Ohio, same with Akron which is located in Ohio and home of Lebron James and Stephen Curry). 

Instead of dealing with all those columns I decided to delete many of them and only focus on a few columns that I think are useful.

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%202.png)

This itself is a good dataframe, but it would be hard to fit a regressor model and find the building occupancy date with the existing columns. To fix this issue I implemented one-hot encoding for every County (and eventually every column) so we could pass it into a machine learning model.

```python
# Convert to lowercase, remove spaces, and strip white spaces
merged_df['County'] = merged_df['County'].str.lower().str.replace(' ', '').str.strip()

# Handle null/NaN and instead call them 'unknown'
merged_df['County'] = merged_df['County'].fillna('unknown')

# One-hot encode the County column
one_hot_encoded_df = pd.get_dummies(merged_df['County'], prefix='', prefix_sep='')

# Rename columns to include '_county' suffix for easy identification 
one_hot_encoded_df.columns = [f"{col}_county" for col in one_hot_encoded_df.columns]

# Add those new encoded columns back into the dataframe 
merged_df = pd.concat([merged_df, one_hot_encoded_df], axis=1)

# Drop the original 'County' column 
merged_df = merged_df.drop(columns=['County'])
```

After running the above command for every other column, I checked how our dataframe was doing.

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%203.png)

Yikes! That’s a ton of memory, plus we still have two columns of object type. I’m going to have to deal with the memory usage of the dataframe soon. For now I focused on the two remaining object type columns.

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%204.png)

In our case our object columns are ‘Bldg Occu Date’ and ‘Int Sq Ft’. Starting with the ‘Int Sq Ft’ column we have NaN values which cause an error when attempting to convert to int.

To handle this I removed the commas and converted each value in the column to numeric, then calculated the median value, and finally replaced each null value with the median value. Finally I converted the entire column to int type.

Building Occupancy Date is a different issue, and it took me a while to find a solution in representing a date as a numeric value. For example just doing MMDDYYYY or DDMMYYYY could cause errors. For example 12/30/2001 and 01/01/2002 are very similar, but converting them directly to a int would make them very far off from each other (12,302,001 vs 1,012,002). 

I vaguely remembered a method for this problem, which was by creating an ‘epoch’ date, which represents time 0. Using this method we can find a numeric way to represent datetime values.

```python
# Step 2: Calculate the number of days since the reference date / Epoch
reference_date = pd.to_datetime('1900-01-01')
merged_df['Bldg Occu Date'] = (merged_df['Bldg Occu Date'] - reference_date).dt.days
```

I choose 1900 as the epoch year because I assumed (and was right) in believing that all existing post offices have been created after the year 1900. 

After implementing this, our dataframe is finally ready to feed into some machine learning models, albeit with some changes needed for each model.

## **3. Data Visualization**

For this histogram showing the distribution of internal square footage, I created it by using seaborn, which is a library that allows for easy creation of visualizations like this. I specified the column name, the number of bins (or boxes) and set ‘kde’ (kernel density estimation) to true (which is an overlay showing the distribution. 

```python
sns.histplot(merged_df['Int Sq Ft'], bins=40, kde=True, color='blue')

```

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%205.png)

This histogram shows that most buildings have small internal square footage, but there seem to be some with extremely large internal square footage (1.75 million square feet!). For a sanity check i went on wikipedia and saw that out of the 3 buildings world wide with over a million square feet floor area, USPS controls none of them. This seems to be a mistake in the data.

I decided to dig deeper into this and created an interactive plot that allows me to zoom in. There seems to be 10 buildings that have 1.75 million square feet, which is clearly a mistake.

In reality the data is heavily skewed towards smaller buildings

(Full zoom)

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%206.png)

(Zoomed in more)

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%207.png)

(Focusing exclusively on the left)

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%208.png)

This data shows us that a majority of our buildings are between 500 to 1000 square feet in size. 

Creating the scatterplot was simple with Seaborn, in which I only had to set the transparency of the plotted points (alpha = 0.6) and choose a color (purple). 

```python
sns.scatterplot(data=merged_df, x='Bldg Occu Date', y='Int Sq Ft', alpha=0.6, color='purple')

```

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%209.png)

This graph shows us that a lot of our buildings are small, and most seem to be created between 1960 and 2000.

I created this graph in a very similar way to the above graphs 

```python
sns.histplot(merged_df['Bldg Occu Date'], bins=40, kde=True, color='blue')
```

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%2010.png)

The number 30,000 represents the year 1983, which seems to be when most of the buildings were created (opened for human occupancy). Out of sheer curiosity I increased the number of bins and replotted the graph.

![image.png](Webscraping%20Regression%2020fdb50af2268045beb7e3db05873df0/image%2011.png)

After settings the parameter bins to 400, it seems that almost all our buildings were open for human occupancy at the same time. This is very interesting.

## 4. Modeling

Yet after all this analysis and data processing, our dataframe now has 19483 rows × 22008 columns. After trying to run a model with this dataframe, my computer started throttling and even after half an hour it didn’t finish. To make our models run, we would need to get rid of a lot of the bloat and useless information in our dataframe.

I visited scikit-learn and found VarianceThreshold which can help us get rid of columns with low variance (i.e. columns that are always True or always False). This can get rid of many of our columns and hopefully improve our attempts of building a machine learning model.

```python
# Reducing our dataset by removing columns that have low variance. (always T always F)
selector = VarianceThreshold(threshold=0.000)
reduced_data = selector.fit_transform(merged_df)
```

After running this code and only including the columns that pass the threshold limit, our columns drop from 22,000 to 103. This will help our models run much quicker, and hopefully be more accurate.

After this preprocessing I attempted to use the following machine learning models:

Linear Regression

Random Forest Regressor
Gradient Boosting Regressor

Gradient Boosting Regressor with RandomizedSearchCV

### Linear Regression

For Linear Regression I didn’t use any hyperparameters and instead used the default implementation from sklearn.

I measured the performance using MAE, MSE, and R-squared which resulted in:

MAE - 3094.09

MSE - 31,125,694.88

R-squared - 0.217

Because of the low R-squared value, it shows that our model struggled due to it’s inability to handle the non-linearity of the data.

### Random Forest Regressor

For Random Forest I used random_state=42, just so I could reproduce results more easily, and set the rest of the parameters to default.

MAE - 2511.54

MSE - 28,464,127.06

R-squared - 0.280

This is slightly better than Linear Regression, which signifies that this model is better at handling non-linear relationships, but without better parameters it struggles.

### Gradient Boosting Regressor

For GBR I set random_state = 42 for reproducibility, but left the other parameters at their default.

MAE - 2699.59

MSE - 25,383,656.15

R-squared - 0.358

This performed much better than Random Forest, but still the default parameters aren’t pushing this model it’s true potential

### Gradient Boosting Regressor with RandomizedSearchCV

This is the same as the above GBR, but instead optimized using hyperparameter tuning with RandomizedSearchCV. I chose randomized search because my computer struggled in searching every possibility, and I also limited it to only searching the following grid (these parameters):

```python
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [30, 5],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 5]
}
```

MAE-2598.26

MSE - 24785840.83

R-squared - 0.373

This is the best performing model so far, and although the hyperparameter tuning is very light (to save compute) it still captures more patterns in the data and performs better.

After using k-fold cross-validation (cv=5) I evaluated the models for mean accuracy and their standard deviation to test overfitting.

| Model | Mean R-squared | Standard Deviation of R-squared |
| --- | --- | --- |
| Linear Regression | 0.217 | 0.008 |
| Random Forest Regressor | 0.280 | 0.011 |
| Gradient Boosting Regressor | 0.358 | 0.009 |
| Gradient Boosting with RandomizedSearchCV | 0.373 | 0.007 |

GBR with RandomizedSearchCV achieved the best average R-squared, and had the lowest standard deviation showing it’s ability to perform well (comparatively) and still avoid overfitting.

Given the significant R-squared gap between models, they’re all statistically significant which is important. 

### Analysis

1. Why do some classifiers work better than others?

In our situation our performance is most significantly tied to how well the models handle non linear relationships and can capture patterns in the data.

Linear Regression assumes a linear relationship, and it’s low R-squared value highlights it’s struggle with capturing complex relationships.

Random Forest performs slightly better, but with default parameters it struggles with both overfitting and making predictions.

Gradient Boosting Regressor performs much better than the first two, which makes sense as it’s best suited for the complex dataset we have, yet the default parameters limit it’s potential.

GBR with RandomizedSearchCV significantly boosts GBR’s performance, and it shows that the model can better adapt to the data. On top of that, the low standard deviation shows that it avoids overfitting (where Random Forest Regressor struggles), yet with the limited parameters it still has a lot of room to improve. 

1. Would another evaluation metric work better than vanilla accuracy?

Yes! R-Squared, MAE, and MSE are more appropriate in our case because we’re performing regression tasks. 

1. Is there still a problem in the data that should be fixed in data cleaning?

Yes! We have many outliers that have been revealed to be extremely different than the median value (internal square footage) and these can cause our models to perform poorly.

1. Does the statistical significance between the different classifiers make sense?

Yes, the statistical significance aligns well with their capabilities. Linear Regression struggles with non-linear relationships, Random Forest struggles with overfitting, and GBR seems the best but requires more hyperparameter tuning.

1. Are there parameters for the classifier that I can tweak to get better performance?

Yes, Gradient Boosting Regressor can have it’s learning_rate, max_depth, and n_estimators tweaked to perform better.

## Conclusion

Our project showcases the ability/capability of machine learning in predicting building occupancy dates (or when it was allowed for human occupancy) based on other information from the building. This can be used together with a report on the difference of USPS buildings and how they’ve changed across the years (and decades). We showed that Gradient Boosting Regressor with RandomizedSearchCV outperforms other models, and that there is room for improvement with further cleaning, feature engineering, and exploration.
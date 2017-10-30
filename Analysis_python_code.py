
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


# In[4]:


df = pd.read_csv("FoodBalanceSheets_E_All_Data.csv", encoding='cp1252')


# In[5]:


df.head(n=30)


# In[6]:


#df2 = pd.DataFrame(np.array(df[['Area','Item','Element','Unit']]))
#df2.columns = ['Area','Item','Element','Unit']

#countries = df['Area'].unique()
#countries

brazil = df[df['Area']=='Brazil']
#print(brazil.head())
russia = df[df['Area']=='Russian Federation']
#print(russia.head())
india = df[df['Area']=='India']
#print(india.head())
china = df[df['Area']=='China']
#print(china.head())
south_africa = df[df['Area']=='South Africa']
#print(south_africa.head())


# In[7]:


prod_brazil = brazil[brazil['Element'] == 'Production']
#print(prod_brazil.head())
prod_russia = russia[russia['Element'] == 'Production']
#print(prod_russia.head())
prod_india = india[india['Element'] == 'Production']
#print(prod_india.head())
prod_china = china[china['Element'] == 'Production']
#print(prod_china.head())
prod_south_africa = south_africa[south_africa['Element'] == 'Production']
#print(prod_south_africa.head())

dem_brazil = brazil[brazil['Element'] == 'Food']
#print(dem_brazil.head())
dem_russia = russia[russia['Element'] == 'Food']
#print(dem_russia.head())
dem_india = india[india['Element'] == 'Food']
#print(dem_india.head())
dem_china = china[china['Element'] == 'Food']
#print(dem_china.head())
dem_south_africa = south_africa[south_africa['Element'] == 'Food']
print(dem_south_africa.head())


# In[8]:


def index_by_year(df):
    year_index = ["Y{year}".format(year=x) for x in range(1961,2014)]
    item_list = ['Cereals - Excluding Beer','Vegetables','Fruits - Excluding Wine','Meat',                  'Fish, Seafood','Eggs','Milk - Excluding Butter']  
    #print(item_list)

    data_table = {} 
    
    for item in item_list:
        year_values = []
        df_entry = df[df['Item'] == item]
    
        [year_values.append(df_entry.iloc[0]['{yr}'.format(yr=year)]) for year in year_index]
        data_table[item] = pd.Series(year_values, index=year_index)
    
    df_by_yr = pd.DataFrame(data_table)
    df_by_yr["Meats"] = df_by_yr["Meat"] + df_by_yr["Eggs"] + df_by_yr["Fish, Seafood"]
    df_by_yr["Fruit+Vegetables"] = df_by_yr["Fruits - Excluding Wine"] + df_by_yr["Vegetables"]
    
    df_by_yr['Unit'] = pd.Series(','.join(['1000 tonnes']*len(year_index)).split(','), index=year_index)
    df_by_yr['Element'] = pd.Series(df['Element'].tolist()[:len(year_index)], index=year_index)
    
    return df_by_yr


# In[9]:


prod_brazil = index_by_year(prod_brazil)
#print(prod_brazil.head())
prod_russia = index_by_year(prod_russia)
#print(prod_russia.head())
prod_india = index_by_year(prod_india)
#print(prod_india.head())
prod_china = index_by_year(prod_china)
print(prod_china.head())
prod_south_africa = index_by_year(prod_south_africa)
#print(prod_south_africa.head())

dem_brazil = index_by_year(dem_brazil)
#print(dem_brazil.head())
dem_russia = index_by_year(dem_russia)
#print(dem_russia.head())
dem_india = index_by_year(dem_india)
#print(dem_india.head())
dem_china = index_by_year(dem_china)
print(dem_china.head())
dem_south_africa = index_by_year(dem_south_africa)
#print(dem_south_africa.head())


# In[10]:


prod_brazil.describe()
prod_russia.describe()
prod_india.describe()
prod_china.describe()
prod_south_africa.describe()


# In[11]:


def pickle_dump(country, prod_df, dem_df):
    prod_df['Year'] = prod_df.index.tolist()
    prod_df = prod_df[['Year','Cereals - Excluding Beer','Milk - Excluding Butter','Meats','Fruit+Vegetables','Unit','Element']]
    dem_df = dem_df[['Cereals - Excluding Beer','Milk - Excluding Butter','Meats','Fruit+Vegetables','Unit','Element']]
    merged_df = pd.merge(dem_df, prod_df, right_index=True, left_index=True)
    print(merged_df.head())
    
    pickle_out = open('{x}.pickle'.format(x=country),'wb')
    pickle.dump(merged_df, pickle_out)
    #pickle_out = open('{x}_dem.pickle'.format(x=country),'wb')
    #pickle.dump(dem_df, pickle_out)
    pickle_out.close()


# In[12]:


pickle_dump('brazil', prod_brazil, dem_brazil)
pickle_dump('russia', prod_russia, dem_russia)
pickle_dump('india', prod_india, dem_india)
pickle_dump('china', prod_china, dem_china)
pickle_dump('south_africa', prod_south_africa, dem_south_africa)


# In[13]:


brazil_data = pd.read_pickle('brazil.pickle')
plt1 = brazil_data.plot()
#plt.legend(loc='Element')
plt.show()


# In[14]:


russia_data = pd.read_pickle('russia.pickle')
plt1 = russia_data.plot()
#plt.legend(loc='Element')
plt.show()


# In[15]:


india_data = pd.read_pickle('india.pickle')
plt1 = india_data.plot()
#plt.legend(loc='Element')
plt.show()


# In[16]:


china_data = pd.read_pickle('china.pickle')
plt1 = china_data.plot()
#plt.legend(loc='Element')
plt.show()


# In[17]:


south_africa_data = pd.read_pickle('south_africa.pickle')
plt1 = south_africa_data.plot()
#plt.legend(loc='Element')
plt.show()


# ####################Linear Regression#####################

# In[18]:


features_df = brazil_data['Year']
features = features_df.as_matrix()
features = np.asarray([float(w[1:]) for w in features])
#features = atures.as_matrix()
features


# In[19]:


target_df = brazil_data['Meats_y']
target = target_df.as_matrix()
target


# In[20]:


from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

regr = linear_model.LinearRegression()


# In[26]:


def get_prediction(country, food_type, f):
    features_df = np.matrix([])
    target_df = np.matrix([])
    if (country == 'brazil'):
        features_df = brazil_data['Year']
        target_df = brazil_data['{x}'.format(x=food_type)]
#    elif (country == 'russia'):
#        features_df = russia_data['Year']
#        target_df = russia_data['{x}'.format(x=food_type)]
    elif (country == 'india'):
        features_df = india_data['Year']
        target_df = india_data['{x}'.format(x=food_type)]
    elif (country == 'china'):
        features_df = china_data['Year']
        target_df = china_data['{x}'.format(x=food_type)]
    elif (country == 'south_africa'):
        features_df = south_africa_data['Year']
        target_df = south_africa_data['{x}'.format(x=food_type)]
    
    #print(country+food_type)
    #print(features_df)
    features = features_df.as_matrix()
    features = np.asarray([float(w[1:]) for w in features])
    #features = atures.as_matrix()
    features
    
    target = target_df.as_matrix()
    target
    
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

#     # # Split the data into training/testing sets
#     features_train = features[:int(0.8*len(features))]
#     features_test = features[int(0.8*len(features)):]

#     # # Split the targets into training/testing sets
#     target_train = target[:int(0.8*len(target))]
#     target_test = target[int(0.8*len(target)):]
        
    #print(features_train.reshape(-1,1))
    #print(target_train.reshape(-1,1))
    
    # Train the model using the training sets
    regr.fit(features_train.reshape(-1, 1), target_train)

    # Make predictions using the testing set
    target_pred = regr.predict(features_test.reshape(-1, 1))
    
    # Plot outputs
    plt.scatter(features_test, target_test,  color='black')
    plt.plot(features_test, target_pred, color='blue', linewidth=3)
    
    #plt.xticks(())
    #plt.yticks(())

    plt.show()
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    f.write('##regression coefficient: '+ str(regr.coef_) + '\n')
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(target_test, target_pred))
    f.write('##regression mean_sq_err: '+ str(mean_squared_error(target_test, target_pred)) + '\n')
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(target_test, target_pred))
    f.write('##variance: '+ str(r2_score(target_test, target_pred)) + '\n')
#    print("Accuracy: "+ str(accuracy_score(target_test, target_pred)) + '\n')
    
    target_pred = regr.predict(np.array([2017]).reshape(1,-1))
    print(target_pred)
    return target_pred


# In[28]:


f = open('reg_results.txt', 'w')
#demand_pred = {}
for country in ['brazil','india','china','south_africa']:
    f.write(country+':\n')

    for food_type in ['Cereals - Excluding Beer_x','Milk - Excluding Butter_x','Meats_x','Fruit+Vegetables_x']:
        prediction = get_prediction(country,food_type,f)
       # if (country == 'china'):
       #     demand_pred.update({food_type: prediction[0][0]})
        f.write('  predicted demand quantity for 2017'+ food_type + ' is ' + str(prediction).strip('[]') + 'x1000 tonnes\n')
    for food_type in ['Cereals - Excluding Beer_y','Milk - Excluding Butter_y','Meats_y','Fruit+Vegetables_y']:
        prediction = get_prediction(country,food_type,f)
        f.write('  predicted production quantity for 2017'+ food_type + ' is ' + str(prediction).strip('[]') + 'x1000 tonnes\n')
f.close()


# In[ ]:


import scipy.stats
from math import sqrt
def mean_confidence_interval(data, confidence=0.90):
    a = 1.0*np.array(data)
    n = len(a)
    mu,sd = np.mean(a),np.std(a)
    z = stats.t.ppf(confidence, n)
    h=z*sd/sqrt(n)
    return mu, std, h


# In[ ]:


print(demand_pred)


# In[ ]:


foods = []
for food_type in demand_pred.keys():
    foods.append(demand_pred[food_type])

mean, std, h = mean_confidence_interval(foods)

for food_type in demand_pred.keys():
    print(food_type+'\n')
    print('  mean: ' + str(mean) + '\n')
    print('  std: ' + str(std) + '\n')
    print('  CI: ' + str(h) + '\n')


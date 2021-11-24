#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from sklearn.linear_model import LogisticRegression
from statsmodels.api import add_constant
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn import metrics
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% ; }</style>"))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


# In[2]:


df = pd.read_csv('data.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.std()


# In[6]:


df['Gender']


# In[7]:


datadict = dict(df['Gender'].value_counts())


# In[8]:


values = list(datadict.values())


# In[9]:


keys = list(datadict.keys())


# In[10]:


df['Gender'].value_counts()


# In[11]:


df['Age'].value_counts()


# In[13]:


column_names = []
for i in df.columns.values:
    column_names.append(i)
column_names


# In[19]:


demographics_list = [
 'Gender',
 'Age',
 'Highest Level of Education:',
 'Number of years in your professional role:',
 'Type of working environment:',
 'How many employees does your company have?']

demographics = df[demographics_list]


# In[20]:


for i in demographics_list:
    print(demographics[i].value_counts())
    print(demographics[i].value_counts()/len(demographics[i]))


# In[21]:


special_demographics_list = [
 'Professional Role:',
 'Please indicate your area of expertise. ',
 'Where is your company located? (Country)',
    
]
special_demographics = df[special_demographics_list]
special_demographics


# In[22]:


df['Professional Role:'].value_counts()


# In[23]:


count_software_developer = 0
count_project_manager = 0
count_uiux = 0
for i in df['Professional Role:']:
    if 'Software Developer' in i:
        count_software_developer+=1
    if 'Project Manager' in i:
        count_project_manager+=1
    if 'UI/UX designer' in i:
        count_uiux+=1
print('count_software_developer', count_software_developer)
print('count_project_manager',count_project_manager)
print('count_uiux',count_uiux)


# In[24]:


roles = []
for i in dict(df['Professional Role:'].value_counts()).keys():
    roles.append(i)
unique_roles = ['Software Developer',
                'UI/UX designer',
                'Project Manager',
                'Graphic designer',
                'Secretary ',
                'Insurance Officer',
                'Teaching ',
                'Cybersecurity Analyst ',
                'Entomologist ',
                'Web Developer ',
                'IT support',
                'digital banking',
                'Digital Marketer'
               ]
unique_roles_dict = []
for i in unique_roles:
    unique_roles_dict.append((i,0))
newdict = dict(unique_roles_dict)
for i in df['Professional Role:']:
    
    for j in unique_roles:
        if j in i:
            newdict[j] = newdict[j]+1
print(newdict)
    
for i in newdict:
    newdict[i] = newdict[i]/107
print(newdict)


# In[25]:


df['Please indicate your area of expertise. '].value_counts().keys()

unique_areas = [
    'Full-stack Development',
    'Front-end Development',
    'Back-end Development',
    'UI/UX',
    'Project Management',
    'Game Development',
    'Cybersecurity',
    'System Administrator',
    'Field and storage research',
    'Research'
]
unique_areas_dict = []
for i in unique_areas:
    unique_areas_dict.append((i,0))
newdict = dict(unique_areas_dict)

for i in df['Please indicate your area of expertise. ']:
    
    for j in unique_areas:
        if j in i:
            newdict[j] = newdict[j]+1
            
print(newdict)
    
for i in newdict:
    newdict[i] = newdict[i]/107
print(newdict)


# In[26]:


count = df['Where is your company located? (Country)'].value_counts()

for i in count:
    print(i/107)
count


# In[27]:


all_questions = ['Projects in our company start with a sponsor user',
 'Sponsor users stay through the project',
 'Sponsor users are very committed throughout the project',
 'Sponsor users have confidence in the project manager and team members',
 'Sponsor users are involved in making schedule estimates and have realistic estimates',
 'We have a project manager on the team for every project',
 'Project managers on projects define the scope for projects with high accuracy',
 'Project managers estimate the cost and time frames for projects accurately',
 'Project managers work as a liaison between the development team and non-development team and gain stakeholders’ approval for project activities.',
 'Project managers provide the development team with all they need to be able to deliver quality products',
 'Project managers are able to measure project progress and control project changes accurately.',
 'Requirements are complete and accurate at the start of the project.',
 'The scope of the project is well defined throughout the project.',
 'Requirements specified at the beginning of the project result in well-defined deliverables',
 'Developers are provided with all the information they need to complete tasks given them.',
 'Internet access is available for every member of the team.',
 'Every single member of the team has access to a personal computer to work with.',
 'Working equipment members use to work are high-level computers that can execute necessary tasks with ease.',
 'Cost of projects satisfy budget.',
 'Time taken to complete project lines up with the schedule',
 'Project specifications and objectives are met.',
 'Projects meet privacy obligations and regulatory compliance',
 ' if applicable.']
sponsorUsers = [
    'Projects in our company start with a sponsor user',
 'Sponsor users stay through the project',
 'Sponsor users are very committed throughout the project',
 'Sponsor users have confidence in the project manager and team members',
 'Sponsor users are involved in making schedule estimates and have realistic estimates'
]

projectManagers = [
    'We have a project manager on the team for every project',
 'Project managers on projects define the scope for projects with high accuracy',
 'Project managers estimate the cost and time frames for projects accurately',
 'Project managers work as a liaison between the development team and non-development team and gain stakeholders’ approval for project activities.',
 'Project managers provide the development team with all they need to be able to deliver quality products',
 'Project managers are able to measure project progress and control project changes accurately.']

requirements = [
    'Requirements are complete and accurate at the start of the project.',
 'The scope of the project is well defined throughout the project.',
 'Requirements specified at the beginning of the project result in well-defined deliverables',
 'Developers are provided with all the information they need to complete tasks given them.',
]

infrastructure = [
 'Internet access is available for every member of the team.',
 'Every single member of the team has access to a personal computer to work with.',
 'Working equipment members use to work are high-level computers that can execute necessary tasks with ease.'
]

success =[
     'Cost of projects satisfy budget.',
 'Time taken to complete project lines up with the schedule',
 'Project specifications and objectives are met.',
 'Projects meet privacy obligations and regulatory compliance',
]


# In[28]:


df[success].value_counts()


# In[29]:


df[sponsorUsers]


# In[30]:


df[projectManagers]


# In[31]:


df[requirements]


# In[32]:


df[infrastructure]


# In[33]:


pg.cronbach_alpha(data=df[success])
len(df[success].columns)


# In[34]:


pg.cronbach_alpha(data=df[sponsorUsers])
len(df[sponsorUsers].columns)


# In[35]:


pg.cronbach_alpha(data=df[projectManagers])
len(df[projectManagers].columns)


# In[36]:


pg.cronbach_alpha(data=df[requirements])
len(df[requirements].columns)


# In[37]:


pg.cronbach_alpha(data=df[infrastructure])
len(df[infrastructure].columns)


# In[38]:


responses = ['Projects in our company start with a sponsor user',
 'Sponsor users stay through the project',
 'Sponsor users are very committed throughout the project',
 'Sponsor users have confidence in the project manager and team members',
 'Sponsor users are involved in making schedule estimates and have realistic estimates',
 'We have a project manager on the team for every project',
 'Project managers on projects define the scope for projects with high accuracy',
 'Project managers estimate the cost and time frames for projects accurately',
 'Project managers work as a liaison between the development team and non-development team and gain stakeholders’ approval for project activities.',
 'Project managers provide the development team with all they need to be able to deliver quality products',
 'Project managers are able to measure project progress and control project changes accurately.',
 'Requirements are complete and accurate at the start of the project.',
 'The scope of the project is well defined throughout the project.',
 'Requirements specified at the beginning of the project result in well-defined deliverables',
 'Developers are provided with all the information they need to complete tasks given them.',
 'Internet access is available for every member of the team.',
 'Every single member of the team has access to a personal computer to work with.',
 'Working equipment members use to work are high-level computers that can execute necessary tasks with ease.',
 'Cost of projects satisfy budget.',
 'Time taken to complete project lines up with the schedule',
 'Project specifications and objectives are met.',
 'Projects meet privacy obligations and regulatory compliance']

for i in responses:
    value = ''
    print(df[i].describe())
    for i in df[i].describe():
        value += str(i) + ','


# In[39]:


def count_responses(col):
    new_dict = {
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0
        }
    for i in df[col]:
        
        new_dict[i] = new_dict[i] + 1
    return new_dict


# In[40]:


df[success]['Cost of projects satisfy budget.']


# In[41]:


for i in responses:
    response = count_responses(i)
    print(i,response)


# In[42]:


df.corr()


# In[43]:


data = {'Customer Involvement': df[sponsorUsers].mean(axis=1),
              'Presence of a Project Manager': df[projectManagers].mean(axis=1),
             'Complete and Accurate Requirements': df[requirements].mean(axis=1),
             'Company and Personal Information Technology Infrastructure':df[infrastructure].mean(axis=1),
            'Software Project Success':df[success].mean(axis=1)
       }


# In[44]:


df_summ = pd.DataFrame(data)


# In[45]:


df_summ.corr()


# In[46]:


x = df_summ[[
    'Customer Involvement', 
         'Presence of a Project Manager',
        'Complete and Accurate Requirements',
        'Company and Personal Information Technology Infrastructure'
            ]]
x = add_constant(x)
y = df_summ['Software Project Success']


# In[47]:


model = OLS(y, x)


# In[48]:


result = model.fit()
print(result.summary())
result.mse_total


# In[49]:


x = df_summ[['Customer Involvement']]
x = add_constant(x)
y = df_summ['Software Project Success']


# In[50]:


model = OLS(y, x)


# In[51]:


result = model.fit()
print(result.params)
print(result.summary())


# In[52]:


result.bse


# In[ ]:





# In[53]:


x = df_summ[[
         'Presence of a Project Manager',
        'Complete and Accurate Requirements',
        'Company and Personal Information Technology Infrastructure'
            ]]
y = df_summ[['Software Project Success']]
print(y.columns)

y['Software Project Success'] = np.where(y['Software Project Success'] > 4, 'Success', 'Failure')


# In[54]:


y.value_counts()


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=50)


# In[ ]:





# In[56]:


logreg = LogisticRegression()

# fit the model with data
logreg.fit(x_train,y_train)


# In[57]:


y_pred=logreg.predict(x_test)


# In[58]:


y_pred


# In[59]:


y_test


# In[60]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)


# In[61]:


cnf_matrix


# In[62]:



logreg.score(x_test,y_test)


# In[63]:


logreg.score(x_train,y_train)


# In[64]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





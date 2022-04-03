import requests

url = 'http://localhost:5000/predict'

input_json_1 = {
    'TotalVisits': 3.0,
    'Total Time Spent on Website': 452,
    'Page Views Per Visit': 3.0,
    'Lead Origin': 'API',
    'Lead Source': 'Organic Search',
    'Do Not Email': 'No',
    'Last Activity': 'Email Opened',
    'Last Notable Activity': 'Email Opened',
    'Specialization': 'Others',
    'What is your current occupation': 'Unemployed',
    'City': 'Mumbai',
    'A free copy of Mastering The Interview': 'No',
    'Lead Quality': 'Not Sure',
    'Tags': 'Ringing'}

input_json_2 = {
    'TotalVisits': 2.0,
    'Total Time Spent on Website': 69,
    'Page Views Per Visit': 2.0,
    'Lead Origin': 'Landing Page Submission',
    'Lead Source': 'Direct Traffic',
    'Do Not Email': 'No',
    'Last Activity': 'SMS Sent',
    'Last Notable Activity': 'SMS Sent',
    'Specialization': 'Human Resource Management',
    'What is your current occupation': 'Unemployed',
    'City': 'Mumbai',
    'A free copy of Mastering The Interview': 'Yes',
    'Lead Quality': 'High in Relevance',
    'Tags': 'Will revert after reading the email'}


for input_json in [input_json_1, input_json_2]:
    r = requests.post(url, json=input_json)
    print(f'Conversion Likelihood: {round(r.json(), 2)}')

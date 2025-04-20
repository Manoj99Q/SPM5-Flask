'''
Goal of Flask Microservice:
1. Flask will take the repository_name such as angular, angular-cli, material-design, D3 from the body of the api sent from React app and 
   will utilize the GitHub API to fetch the created and closed issues. Additionally, it will also fetch the author_name and other 
   information for the created and closed issues.
2. It will use group_by to group the data (created and closed issues) by month and will return the grouped data to client (i.e. React app).
3. It will then use the data obtained from the GitHub API (i.e Repository information from GitHub) and pass it as a input request in the 
   POST body to LSTM microservice to predict and forecast the data.
4. The response obtained from LSTM microservice is also return back to client (i.e. React app).

Use Python/GitHub API to retrieve Issues/Repos information of the past 1 year for the following repositories:
- https: // github.com/angular/angular
- https: // github.com/angular/material
- https: // github.com/angular/angular-cli
- https: // github.com/d3/d3
'''
# Import all the required packages 
import os
from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
import json
import dateutil.relativedelta
from dateutil import *
from datetime import date
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)

# Add response headers to accept all types of  requests
def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

# Modify response headers when returning to the origin
def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
Health check endpoint
This endpoint is used for health monitoring
'''
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "UP",
        "service": "Flask GitHub API Service",
        "timestamp": str(date.today())
    }), 200

'''
API route path is  "/api/github"
This API will accept only POST request
'''
@app.route('/api/github', methods=['POST'])
def github():
    body = request.get_json()
    # Extract the choosen repositories from the request
    repo_name = body['repository']
    # Extract the data type from the request (issues or pulls)
    data_type = body.get('dataType', 'issues')  # Default to issues if not specified
    
    # Add your own GitHub Token to run it local
    token = os.environ.get(
        'GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')
    GITHUB_URL = f"https://api.github.com/"
    headers = {
        "Authorization": f'token {token}'
    }
    params = {
        "state": "all"  # Get all issues/PRs regardless of state
    }
    repository_url = GITHUB_URL + "repos/" + repo_name
    # Fetch GitHub data from GitHub API
    repository = requests.get(repository_url, headers=headers)
    # Convert the data obtained from GitHub API to JSON format
    repository = repository.json()

    today = date.today()
    
    # Initialize response data
    created_at_issues = []
    closed_at_issues = []
    pulls_data = []
    created_at_image_urls = {}
    closed_at_image_urls = {}
    pulls_image_urls = {}

    # Check if we're in development environment
    IS_DEV_ENV = os.environ.get('FLASK_ENV', '') == 'development'
    
    # Use local LSTM service in dev environment, otherwise use cloud URL
    if IS_DEV_ENV:
        LSTM_API_URL = "http://lstm-service:8080/api/forecast"
    else:
        # Update your Google cloud deployed LSTM app URL (NOTE: DO NOT REMOVE "/")
        LSTM_API_URL = os.environ.get("LSTM_API_URL", "https://forecast-service-852131999673.us-central1.run.app/") + "api/forecast"

    # Process based on data type requested
    if data_type == 'issues':
        # Fetch and process only issues data
        issues_response = fetch_github_data(repo_name, today, headers, params, 'issue')
        
        # Process issues data
        df_issues = pd.DataFrame(issues_response)
        
        # Format issues data for frontend
        created_at_issues, closed_at_issues = format_github_data(df_issues) if not df_issues.empty else ([], [])
        
        # Prepare data for LSTM forecasting
        created_at_body = {
            "issues": issues_response,
            "type": "created_at",
            "repo": repo_name.split("/")[1]
        }
        closed_at_body = {
            "issues": issues_response,
            "type": "closed_at",
            "repo": repo_name.split("/")[1]
        }
        
        # Get forecasts for created issues
        created_at_response = requests.post(LSTM_API_URL,
                                           json=created_at_body,
                                           headers={'content-type': 'application/json'})
        
        # Get forecasts for closed issues
        closed_at_response = requests.post(LSTM_API_URL,
                                         json=closed_at_body,
                                         headers={'content-type': 'application/json'})
                                         
        # Store responses                                 
        created_at_image_urls = created_at_response.json()
        closed_at_image_urls = closed_at_response.json()
        
    elif data_type == 'pulls':
        # Fetch and process only pull requests data
        pulls_response = fetch_github_data(repo_name, today, headers, params, 'pr')
        
        # Process pull requests data
        df_pulls = pd.DataFrame(pulls_response)
        
        # Format pull requests data for frontend
        pulls_data = format_pulls_data(df_pulls) if not df_pulls.empty else []
        
        # For pull requests, modify the data structure for LSTM service compatibility
        pulls_for_lstm = []
        for pull in pulls_response:
            pull_modified = pull.copy()
            pull_modified['created_at'] = pull['created_at']
            pulls_for_lstm.append(pull_modified)
        
        pulls_body = {
            "issues": pulls_for_lstm,
            "type": "created_at",
            "repo": repo_name.split("/")[1] + "_pulls"
        }
        
        # Get forecasts for pull requests
        try:
            pulls_response_forecast = requests.post(LSTM_API_URL,
                                                  json=pulls_body,
                                                  headers={'content-type': 'application/json'})
            pulls_image_urls = pulls_response_forecast.json()
        except Exception as e:
            print(f"Error getting pull request forecasts: {str(e)}")
            pulls_image_urls = {
                "model_loss_image_url": "",
                "lstm_generated_image_url": "",
                "all_issues_data_image": ""
            }

    # Create response with the requested data
    json_response = {
        "created": created_at_issues,
        "closed": closed_at_issues,
        "pulls": pulls_data,
        "starCount": repository["stargazers_count"],
        "forkCount": repository["forks_count"],
        "createdAtImageUrls": created_at_image_urls,
        "closedAtImageUrls": closed_at_image_urls,
        "pullsImageUrls": pulls_image_urls,
    }
    
    # Return the response back to client (React app)
    return jsonify(json_response)

'''
Helper function to fetch GitHub data (issues or pull requests)
'''
def fetch_github_data(repo_name, today, headers, params, data_type):
    GITHUB_URL = "https://api.github.com/"
    response_data = []
    
    # Iterating to get data for every month for the past 12 months
    for i in range(12):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        
        if data_type == 'issue':
            types = 'type:issue'
        else:
            types = 'type:pr'
            
        repo = 'repo:' + repo_name
        ranges = 'created:' + str(last_month) + '..' + str(today)
        per_page = 'per_page=100'
        
        # Search query will create a query to fetch data for a given repository in a given time range
        search_query = types + ' ' + repo + ' ' + ranges

        # Append the search query to the GitHub API URL 
        query_url = GITHUB_URL + "search/issues?q=" + search_query + "&" + per_page
        
        # requsets.get will fetch requested query_url from the GitHub API
        search_results = requests.get(query_url, headers=headers, params=params)
        
        # Convert the data obtained from GitHub API to JSON format
        search_results = search_results.json()
        results_items = []
        
        try:
            # Extract "items" from search results
            results_items = search_results.get("items")
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
            
        if results_items is None:
            continue
            
        for item in results_items:
            label_name = []
            data = {}
            current_item = item
            
            # Get issue/PR number
            data['issue_number'] = current_item["number"]
            
            # Get created date
            data['created_at'] = current_item["created_at"][0:10]
            
            if current_item["closed_at"] == None:
                data['closed_at'] = current_item["closed_at"]
            else:
                # Get closed date
                data['closed_at'] = current_item["closed_at"][0:10]
                
            for label in current_item["labels"]:
                # Get label name
                label_name.append(label["name"])
                
            data['labels'] = label_name
            
            # It gives state like closed or open
            data['State'] = current_item["state"]
            
            # Get Author
            data['Author'] = current_item["user"]["login"]
            
            # Add a flag to identify if it's a pull request
            data['is_pull_request'] = 'pull_request' in current_item
            
            response_data.append(data)

        today = last_month
        
    return response_data

'''
Helper function to format GitHub issues data for the frontend
'''
def format_github_data(df):
    if df.empty:
        return [], []
        
    # Daily Created Issues
    df_created_at = df.groupby(['created_at'], as_index=False).count()
    dataFrameCreated = df_created_at[['created_at', 'issue_number']]
    dataFrameCreated.columns = ['date', 'count']

    # Monthly Created Issues
    created_at = df['created_at']
    month_issue_created = pd.to_datetime(
        pd.Series(created_at), format='%Y-%m-%d')
    month_issue_created.index = month_issue_created.dt.to_period('m')
    month_issue_created = month_issue_created.groupby(level=0).size()
    month_issue_created = month_issue_created.reindex(pd.period_range(
        month_issue_created.index.min(), month_issue_created.index.max(), freq='m'), fill_value=0)
    month_issue_created_dict = month_issue_created.to_dict()
    created_at_issues = []
    for key in month_issue_created_dict.keys():
        array = [str(key), month_issue_created_dict[key]]
        created_at_issues.append(array)

    # Monthly Closed Issues
    closed_at = df['closed_at'].sort_values(ascending=True)
    # Filter out None values
    closed_at = closed_at[closed_at.notnull()]
    
    if len(closed_at) > 0:
        month_issue_closed = pd.to_datetime(
            pd.Series(closed_at), format='%Y-%m-%d')
        month_issue_closed.index = month_issue_closed.dt.to_period('m')
        month_issue_closed = month_issue_closed.groupby(level=0).size()
        month_issue_closed = month_issue_closed.reindex(pd.period_range(
            month_issue_closed.index.min(), month_issue_closed.index.max(), freq='m'), fill_value=0)
        month_issue_closed_dict = month_issue_closed.to_dict()
        closed_at_issues = []
        for key in month_issue_closed_dict.keys():
            array = [str(key), month_issue_closed_dict[key]]
            closed_at_issues.append(array)
    else:
        closed_at_issues = []
        
    return created_at_issues, closed_at_issues

'''
Helper function to format pull requests data for the frontend
'''
def format_pulls_data(df):
    if df.empty:
        return []
        
    # Filter only pull requests
    df_pulls = df[df['is_pull_request'] == True]
    
    if df_pulls.empty:
        return []
        
    # Monthly Pull Requests
    created_at = df_pulls['created_at']
    month_pulls_created = pd.to_datetime(
        pd.Series(created_at), format='%Y-%m-%d')
    month_pulls_created.index = month_pulls_created.dt.to_period('m')
    month_pulls_created = month_pulls_created.groupby(level=0).size()
    month_pulls_created = month_pulls_created.reindex(pd.period_range(
        month_pulls_created.index.min(), month_pulls_created.index.max(), freq='m'), fill_value=0)
    month_pulls_created_dict = month_pulls_created.to_dict()
    pulls_data = []
    for key in month_pulls_created_dict.keys():
        array = [str(key), month_pulls_created_dict[key]]
        pulls_data.append(array)
        
    return pulls_data

# Run flask app server on port 5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

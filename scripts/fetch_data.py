import requests
import json

def fetch_match_ids(lang,team_id):
    url = "https://pp-hs-consumer-api.espncricinfo.com/v1/pages/matches/result?lang="+lang
    headers = {
        "x-hsci-auth-key": "",
        "accept": "application/json"
    }
    response = requests.get(url,headers=headers)
    data = json.loads(response.text)
    result = []
    print(len(data["content"]["matches"]))

    for match in data["content"]["matches"]:
        for team_data in match["teams"]:
            if team_data["team"]["id"] == team_id:
                result.append([match["objectId"], match["series"]["objectId"]])
    return result

def fetch_commentary(lang,match_info):
    result = []
    for match_id, series_id in match_info:
        url = f'https://pp-hs-consumer-api.espncricinfo.com/v1/pages/match/commentary?lang={lang}&seriesId={series_id}&matchId={match_id}&sortDirection=DESC'
        headers = {
                "x-hsci-auth-key": "",
                "accept": "application/json"
        }
        response = requests.get(url,headers=headers)
        data = json.loads(response.text)
        match_details = {}
        match_details["series_id"]=series_id
        match_details["match_id"]=match_id
        match_details["comments"] = (data["content"]["comments"])
        match_details["overComments"] = (data["content"]["overComments"])
        match_details["recentOverCommentary"] = (data["content"]["recentOverCommentary"])
        match_details["recentBallCommentary"] = (data["content"]["recentBallCommentary"])

        result.append(match_details)
    return result

match_id_series_id = fetch_match_ids("en",6)

english_comment = fetch_commentary("en",match_id_series_id)
# Serializing json
json_object = json.dumps(english_comment, indent=4)
 
# Writing to sample.json
with open("english_commentary.json", "w") as outfile:
    outfile.write(json_object)

hindi_comment = fetch_commentary("hi",match_id_series_id)
json_hindi_object = json.dumps(hindi_comment, indent=4)
 
# Writing to sample.json
with open("hindi_commentary.json", "w") as outfile:
    outfile.write(json_hindi_object)
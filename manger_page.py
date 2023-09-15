#imported all necessary modules for projecting "data engineer" cadidate list to the manager in a web page
from flask import Flask,jsonify, request,render_template,session
from DecisionTree_working_code import filtered_data_enginner_df_final
app = Flask(__name__)

#rested the indexes 
filtered_data_enginner_df_final=filtered_data_enginner_df_final.reset_index(drop=True)

#created a route for the web application
@app.route("/dataengineer",methods=("POST","GET"))
def test():
    return render_template('dataengineer.html',tables=[filtered_data_enginner_df_final.to_html(classes='data')], titles=filtered_data_enginner_df_final.columns.values)

if __name__=="__main__":
   try:
        app.run(host="127.0.0.1", port=5000, debug=True,use_reloader=False)
        
   except Exception as e:
       print(e)
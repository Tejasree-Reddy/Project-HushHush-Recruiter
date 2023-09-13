import pandas as pd
import threading
import multiprocessing
from parsel import Selector
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from threading import Lock

opts = Options()

driver = webdriver.Chrome(options=opts, executable_path=r'C:\\Users\\dell\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe')

# def validate_field(field):
#     if field:
#         pass
#     else:
#         field='No results'
#     return field

driver.get('https://www.linkedin.com/login?fromSignIn=true&trk=guest_homepage-basic_nav-header-signin')
username = driver.find_element(By.ID,'username')
username.send_keys('rashmignaik231@gmail.com')
sleep(0.5)
password = driver.find_element(By.ID,'password')
password.send_keys('Rashmi@23')
sleep(0.5)
sign_in_button = driver.find_element(By.XPATH,'//*[@type="submit"]')
sign_in_button.click()
sleep(20)

profile_data=[]
links =[]

# href_skill = [element.get_attribute("href") for element in driver.find_elements(By.XPATH, "//div[@class='pvs-list__footer-wrapper']//a[contains(@id,'skills')]")]
# href_exp = [element.get_attribute("href") for element in driver.find_elements(By.XPATH, "//div[@class='pvs-list__footer-wrapper']//a[contains(@id,'experiences')]")]

def education(prof_link):
    driver.get(prof_link)
    sleep(6)
    sel= Selector(text=driver.page_source)
    
    try:
        edu = sel.xpath('//span[contains(@class, "t-14 t-normal")]/span[contains(@aria-hidden,"true")]/text()')
    except:
        edu = 'None'
    edu_list =[]
    for e in edu:
        edu_list.append(str(e))

    educ = []

    for e in edu_list:
        if "Master" in e or "Bachelor" in e or "PhD" in e or "Data" in e:
            educ.append(e[109:-2])
        else:
            None
    
    # print(education_list)
    return educ

def experience(prof_link,href_exp=[None]):
    sel= Selector(text=driver.page_source)
    if href_exp:
        for e in href_exp:
            driver.get(e)
            sleep(3)
            exp = sel.xpath('//span[contains(@class,"t-14 t-normal t-black--light")]/span[contains(@aria-hidden,"true")]/text()')
    else:
        exp = sel.xpath('//span[contains(@class,"t-14 t-normal t-black--light")]/span[contains(@aria-hidden,"true")]/text()')
        
    exp_temp =[]
    for j in exp:
        exp_temp.append(str(j).split('·'))
    experience_list = []
    experiences =[]
    for x in range(0,len(exp_temp),2):
        if len(exp_temp[x])>1:
            experience_list.append(exp_temp[x][1][0:-2])
    cum_exp = 0
    for m in experience_list:
        experiences = [int(x) for x in m.split() if x.isnumeric()]
        
        if len(experiences) >= 2:
            cum_exp += 12 * experiences[0] + experiences[1]
        elif len(experiences) == 1 and "yr" in m:
            cum_exp += 12 * experiences[0]
        elif len(experiences) == 1 and "mo" in m:
            cum_exp += experiences[0]
    
    return cum_exp        
    
#     for m in experience_list:
#     if "yr" in m or "mo" in m:
#         if len(m)==12 or len(m)==13 or len(m)==14:
#             year=int(m[0:2])*12+int(m[6:8])
#         elif "yr" in m:
#             if len(m)==5 or len(m)==6:
#                 year=int(m[0:2])*12
#         else:
#             year = int(m[0:2])
        
#         experiences.append(year)
# #         experiences= sum(int(experiences))
#     else:
#         None
# experiences = sum(experiences)
                   
#     # print(experiences)
#     return experiences


def skills(prof_link,href_skill=[None]):
    skills_list = []
    if href_skill:
        for skill in href_skill:
            driver.get(skill)
            sleep(4)
            sel= Selector(text=driver.page_source)
            whole_skills_data = sel.xpath('//div[contains(@class,"mr1 hoverable-link-text t-bold")]/span[contains(@aria-hidden,"true")]/text()')
            for i in whole_skills_data:
                skills_list.append(str(i)[124:-2])
    else:
        skills_list =[]
    
    
    
    return list(set(skills_list))

for x in range(0, 300, 10):
    url = f'https://www.google.com/search?q=site:linkedin.com/in/+AND+%22Data+Analyst%22&sca_esv=563362597&sxsrf=AB5stBhLAVFfbnVngKXZmGNEATDJC4W1DQ:1694092830048&ei=Hs75ZPXJApL0sAfWy4uYCA&start={x}'
    driver.get(url)
    # Wait for the search results to load
    WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.XPATH, "//div[@class='yuRUbf']//a")))
    # Scrape the LinkedIn URLs
    linkedin_urls = [my_element.get_attribute("href") for my_element in driver.find_elements(By.XPATH, "//div[@class='yuRUbf']//a")]
    links.extend(linkedin_urls)
    # Append the URLs to the links list
    # Wait for a short interval before the next iteration
    sleep(20)
# len(links)

# removing duplicates from the urls list and filtering the unwanted urls
data = list(set(links))
filtered_links = [l for l in data if "translate.google.com/translate?" not in l]
# print(filtered_links)
# print(links)    
split_links= [filtered_links[x:x+50] for x in range(0,len(filtered_links), 50)]


lock=Lock()

ID=20230000    
def linkedin_scrape_function(chunk,lock):
    
    for link in chunk:
        lock.acquire()
        driver.get(link)
        sleep(8)
        sel= Selector(text=driver.page_source)
        
        name = sel.xpath('//*[starts-with(@class,"text-heading-xlarge inline t-24 v-align-middle break-words")]/text()').extract_first()
        job_title = sel.xpath('//*[starts-with(@class,"text-body-medium break-words")]/text()').extract_first()
        if name and job_title:
            name = name.strip()
            job_title = job_title.strip()
        skill_url = []
        exp_url=[]
        href_skill = [element.get_attribute("href") for element in driver.find_elements(By.XPATH, "//div[@class='pvs-list__footer-wrapper']//a[contains(@id,'skills')]")]
        skill_url.extend(href_skill)
        href_exp = [element.get_attribute("href") for element in driver.find_elements(By.XPATH, "//div[@class='pvs-list__footer-wrapper']//a[contains(@id,'experiences')]")]
        exp_url.extend(href_exp)
        
        candidate_edu=education(link)

        candidate_exp=experience(link,exp_url)
        # print(href_skill)
        candidate_skills=skills(link,skill_url)
        ID=ID+1


        
        data={'ID':ID,
            'Name':name,
            'Job_Title':job_title,
            'education':candidate_edu,
            'expereince':candidate_exp,
            'Skills':candidate_skills}

        profile_data.append(data)
        lock.release()
    return profile_data

threadList = []
for i in range(len(split_links)):
    t = threading.Thread(target=linkedin_scrape_function, args=[split_links[i],lock])
#     multiprocessing.Process(tar)
    t.start()
    threadList.append(t)
# print(threadList)

for eachThread in threadList:
    eachThread.join()
    
df= pd.DataFrame(profile_data)

df.to_excel("Linkedin_dataset.xlsx")

driver.quit()
import cv2
import typing
import oracledb
import numpy as np
from csv import writer
from selenium import webdriver
from selenium.webdriver.common.by import By
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from selenium.webdriver.chrome.options import Options

v_process_user = 'U1'

# OCR model to decode CAPTCHA
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, (self.input_shape[2], self.input_shape[1]))  
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

def Submit():
    Captchaimg = driver.find_element(By.ID, "capimage")
    driver.execute_script("arguments[0].scrollIntoView(true);", Captchaimg)
    Captchaimg.screenshot('Screenshotcaptcha.png')

    configs = BaseModelConfigs.load("Models/02_captcha_to_text/202502191616/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    image = cv2.imread('Screenshotcaptcha.png')
    prediction_text = model.predict(image)

    i_Captcha = driver.find_element(By.ID, "Captcha1")
    i_Captcha.clear()     
    i_Captcha.send_keys(prediction_text)
    
    Submit_button = driver.find_element(By.ID, 'Submit')
    driver.execute_script("arguments[0].click();", Submit_button)

# Output Data
def Data_Xpath_Elements():
    v_AppNo = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[1]/td[2]').text
    v_RollNo_SESS1 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[1]/td[4]').text
    v_RollNo_SESS2 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[1]/td[6]').text	
    v_CName = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[2]/td[2]').text
    v_MName = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[3]/td[2]').text
    v_FName = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[4]/td[2]').text
    v_Category = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[5]/td[2]').text
    v_PwBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[5]/td[4]').text
    v_Gender = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[6]/td[2]').text
    v_DoB = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[6]/td[4]').text
    v_StateofEligibility = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[7]/td[2]').text
    v_Nationality = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[2]/td/table/tbody/tr[7]/td[4]').text

    v_Physics_SESS1 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[4]/td[2]').text
    v_Chemistry_SESS1 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[5]/td[2]').text
    v_Mathematics_SESS1 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[3]/td[2]').text
    v_Total_SESS1 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[6]/td[2]').text

    v_Physics_SESS2 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[4]/td[3]').text
    v_Chemistry_SESS2 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[5]/td[3]').text
    v_Mathematics_SESS2 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[3]/td[3]').text
    v_Total_SESS2 = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[6]/td[3]').text

    v_Physics_RES = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[4]/td[4]').text
    v_Chemistry_RES = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[5]/td[4]').text
    v_Mathematics_RES = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[3]/td[4]').text
    v_Total_RES = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[6]/td[4]').text

    v_TotinWords = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[4]/td/table/tbody/tr[7]/td[2]').text

    v_DoR = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[8]/td/table/tbody/tr[3]/td/strong').text.replace("Date : ", "")

    v_course = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/th').text
    v_CRL = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[1]').text
    v_GEN_EWS = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[2]').text
    v_OBC_NCL = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[3]').text
    v_SC = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[4]').text
    v_ST = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[5]').text

    v_PwBD_CRL = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[6]').text
    v_PwBD_GEN_EWS = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[7]').text
    v_PwBD_OBC_NCL = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[8]').text
    v_PwBD_SC = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[9]').text
    v_PwBD_ST = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[10]').text

    v_Remarks = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[8]/td/table/tbody/tr[1]/td').text.replace("Remarks : ", "")

    insert_stu_dtls = \
                "INSERT INTO O_JEEMAIN_RESULT_SESS2_25 (ADMNO, APPNO, ROLLNO_SESS1, ROLLNO_SESS2, CANDIDATE_NAME, DOB, GENDER, MOTHER, FATHER, CATEGORY, PWBD, STATEOFELIGIBILITY, NATIONALITY, \
                MATHEMATICS_SESS1, PHYSICS_SESS1, CHEMISTRY_SESS1, TOTAL_SESS1, MATHEMATICS_SESS2, PHYSICS_SESS2, CHEMISTRY_SESS2, TOTAL_SESS2 , MATHEMATICS_FINAL, PHYSICS_FINAL, CHEMISTRY_FINAL , \
                TOTAL_FINAL, TOTAL_IN_WORDS, DATEOFRESULT, COURSE, CRL, GEN_EWS, OBC_NCL, SC,ST,PWBD_CRL, PWBD_GEN_EWS, PWBD_OBC_NCL, PWBD_SC, PWBD_ST, REMARKS, \
                PROCESS_STATUS, PROCESS_USER) \
                VALUES ('0', '" + str(v_AppNo) + "', '" + str(v_RollNo_SESS1) + "', '" + str(v_RollNo_SESS2) + "', '" + str(v_CName) + "', '" + str(v_DoB) + "', '" + str(v_Gender) + "', '" + str(v_MName) + "', '" + str(v_FName) + "', \
                 '" + str(v_Category) + "', '" + str(v_PwBD) + "', '" + str(v_StateofEligibility) + "', '" + str(v_Nationality) + "', '" + v_Mathematics_SESS1 + "', '" + v_Physics_SESS1 + "', '" + v_Chemistry_SESS1 + "', '" + v_Total_SESS1 + "', \
                '" + v_Mathematics_SESS2 + "', '" + v_Physics_SESS2 + "', '" + v_Chemistry_SESS2 + "', '" + v_Total_SESS2 + "', '" + v_Mathematics_RES + "', '" + v_Physics_RES + "', '" + v_Chemistry_RES + "', '" + v_Total_RES + "', '" + str(v_TotinWords) + "', '" + v_DoR + "',\
                '" + v_course + "', '" + v_CRL + "', '" + v_GEN_EWS + "', '" + v_OBC_NCL + "', '" + v_SC + "', '" + v_ST + "', '" + v_PwBD_CRL + "', '" + v_PwBD_GEN_EWS + "', '" + v_PwBD_OBC_NCL + "', '" + v_PwBD_SC + "', '" + v_PwBD_ST + "', '" + v_Remarks + "' , 'D', '" + str(v_process_user) + "')"      
    cur.execute(insert_stu_dtls)

    update_IpStatus = "UPDATE I_JEEMAIN_RESULT_SESS2_25 SET PROCESS_STATUS = 'Done', PROCESS_USER = '"+ str(v_process_user) +"', CREATED_DATE = SYSDATE WHERE APPNO = '"+ str(v_appno) +"'"
    cur.execute(update_IpStatus)
    conn.commit()
    
max_attempts = 20
attempts = 0
login_successful = False
    
#oracledb.init_oracle_client()
oracledb.init_oracle_client(lib_dir=r"D:\app\udaykumard\product\instantclient_23_6")
conn = oracledb.connect(user='RESULT', password='LOCALDEV', dsn='192.168.15.196:1521/orcldev')
cur = conn.cursor()

# Start Chrome
chrome_options = Options()
#chrome_options.add_argument("--headless")  # Remove this if you want to see browser
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(options=chrome_options)
driver.maximize_window()

str_dataslot = "SELECT PROCESS_USER, START_VAL, END_VAL FROM DATASLOTS_VAL_USER WHERE PROCESS_USER = '"+v_process_user+"'"
cur.execute(str_dataslot)
res_dataslot = cur.fetchall()

start_sno = res_dataslot[0][1]
end_sno = res_dataslot[0][2]

Sno = start_sno

str_Jeeappno = "SELECT SNO, TRIM(APPNO) APPNO, TRIM(PASSWORD) PASSWORD, ADMNO FROM I_JEEMAIN_RESULT_SESS2_25 \
            WHERE PROCESS_STATUS = 'P' AND PASSWORD IS NOT NULL AND LENGTH(APPNO) = 12 AND \
            SNO >= '"+str(start_sno)+"' AND  SNO <='"+str(end_sno)+"' ORDER BY SNO"
cur.execute(str_Jeeappno)
res = cur.fetchall()

for row in res:
    v_appno = row[1]     
    v_password = row[2]
    v_AdmNo = row[3]
            
    try:  
        driver.get("https://cnr.nic.in/ResultDir/JEEMAIN2025S2P1/Login")
        
        i_regno = driver.find_element(By.ID, "txtAppNo").send_keys(v_appno)      
        i_password = driver.find_element(By.ID, "txtPassword").send_keys(v_password)
        Submit()
            
        try:
            v_invalid = driver.find_element(By.XPATH, '/html/body/form/div[2]/div[2]/div/fieldset/div/div/div[6]/div[2]/span').text 
        except:
            try:
                v_invalid = driver.find_element(By.XPATH, '/html/body/form/div[2]/div[2]/div/fieldset/div/div/div[6]/div[2]/span').text
            except:
                v_invalid = '' 

        if v_invalid == "Invalid CAPTCHA":
            while attempts < max_attempts and not login_successful:
                Submit()

                try:
                    Data_Xpath_Elements()
                    login_successful = True  
                except:
                    try:
                        v_invalid = driver.find_element(By.XPATH, '/html/body/form/div[2]/div[2]/div/fieldset/div/div/div[6]/div[2]/span').text 
                    except:
                        try:
                            v_invalid = driver.find_element(By.XPATH, '/html/body/form/div[2]/div[2]/div/fieldset/div/div/div[6]/div[2]/span').text
                        except:
                            v_invalid = '' 

                    if v_invalid == "Invalid Application Number/Password":        
                        update_IpStatus = "UPDATE I_JEEMAIN_RESULT_SESS2_25 SET PROCESS_STATUS = 'Invalid', PROCESS_USER = '"+ str(v_process_user) +"', CREATED_DATE = SYSDATE, ERROR_MESSAGE = '"+ str(v_invalid) +"' WHERE APPNO = '"+ str(v_appno) +"'"
                        cur.execute(update_IpStatus)
                        conn.commit()
                        login_successful=True
                    else:
                        login_successful=False

        elif v_invalid == "Invalid Application Number/Password":        
            update_IpStatus = "UPDATE I_JEEMAIN_RESULT_SESS2_25 SET PROCESS_STATUS = 'Invalid', PROCESS_USER = '"+ str(v_process_user) +"', CREATED_DATE = SYSDATE, ERROR_MESSAGE = '"+ str(v_invalid) +"' WHERE APPNO = '"+ str(v_appno) +"'"
            cur.execute(update_IpStatus)
            conn.commit()
            pass
        else:
            Data_Xpath_Elements()
            login_successful = True
    except:
        pass

driver.quit()
                                    



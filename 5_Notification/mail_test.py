#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 23:04:42 2021

@author: Aron
"""


# % 讀取套件 -------
import pandas as pd
import numpy as np
import sys, time, os, gc

# import smtplib
# from email import encoders
# from email.mime.base import MIMEBase
# from email.mime.multipart import MIMEMultipart


local = False
local = True


# Path .....
if local == True:
    path = '/Users/Aron/Documents/GitHub/Data/Stock_Analysis'
else:
    path = '/home/aronhack/stock_forecast/dashboard'
    # path = '/home/aronhack/stock_analysis_us/dashboard'


# Codebase ......
path_codebase = [r'/Users/Aron/Documents/GitHub/Arsenal/',
                 r'/Users/Aron/Documents/GitHub/Codebase_YZ']


for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path


import codebase_yz as cbyz
import arsenal as ar


# 自動設定區 -------
pd.set_option('display.max_columns', 30)
 




ar.send_mail(to='myself20130612@gmail.com', subject='Knock Knock', 
          content='Test Mail', preamble='ARON HACK Mail')



sender = 'aronhack.noreply@gmail.com'
gmail_password = '!BUvw5Z]k;gz(EbW*Nr='

df = pd.DataFrame({'EMAIL':[sender],
                   'PWD':[gmail_password]})


df.to_csv('/Users/Aron/Documents/email_info.csv', index=False)






# sender = 'aronhack.noreply@gmail.com'
# gmail_password = '2021zyamiya'

# recipients_list = ['myself20130612@gmail.com']
# recipients = cbyz.li_join_as_str(recipients_list)

# outer = MIMEMultipart()
# outer['Subject'] = 'Test Mail'
# outer['To'] = recipients
# outer['From'] = sender
# outer.preamble = 'You will not see this in a MIME-aware	mail reader.\n'
# 	


# # # Add attachments				
# # attachments = [dest_filename]
# # 				
# # # 檔案到MAIL底下
# # for file in attachments:
# #     try:
# #         with open(file, 'rb') as fp:
# #             print ('can read faile')
# #             msg = MIMEBase('application', "octet-stream")
# #             msg.set_payload(fp.read())
# #             encoders.encode_base64(msg)
# #             msg.add_header('Content-Disposition', 'attachment', filename = os.path.basename(file))
# #             outer.attach(msg)
# #     except:
# #         print("Unable to open one of the attachments. Error: ",	sys.exc_info()[0])
# #         raise


# composed = outer.as_string()
# try:
#     with smtplib.SMTP('smtp.gmail.com',	587) as s:
#         s.ehlo()
#         s.starttls()
#         s.ehlo()
#         s.login(sender,	gmail_password)
#         s.sendmail(sender,	recipients,	composed)
#         s.close()
#         print("Email sent!")
# except:
#     print("Unable to send the email. Error: ",	sys.exc_info()[0])
#     raise
# 		    





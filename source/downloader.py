import os
import io
import shutil
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
if not os.path.exists('token.json'):
    f = open('token.json','w')
    f.write('[TOKEN]')
    f.close()
SCOPES = ['https://www.googleapis.com/auth/drive']
try:
    creds = Credentials.from_authorized_user_file('token.json',SCOPES)
except:
    creds = False
if not creds or creds.expired:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('token.json',SCOPES)
        creds = flow.run_console()
    f = open('token.json','w')
    f.write(creds.to_json())
    f.close()
service = build('drive','v3',credentials=creds)

def googleGetIdOfPath(path, startid='root'):
    if path[0] == '/':  path = path[1:]
    path = path.split('/')
    idxs = [startid]
    for i in range(len(path)):
        results = service.files().list(q="'{}' in parents and trashed=false and name='{}'".format(idxs[i],path[i]),includeItemsFromAllDrives=True,supportsAllDrives=True).execute()
        idxs.append(results['files'][0]['id'])
    idxs = idxs[1:]
    return idxs[-1]

def googleListFilesAtPath(path, startid='root'):
    idx = googleGetIdOfPath(path, startid=startid)
    results = service.files().list(q="'{}' in parents and trashed=false".format(idx),includeItemsFromAllDrives=True,supportsAllDrives=True).execute()
    files = results['files']
    while 'nextPageToken' in results.keys():
        results = service.files().list(q="'{}' in parents and trashed=false".format(idx),pageToken=results['nextPageToken'],includeItemsFromAllDrives=True,supportsAllDrives=True).execute()
        files += results['files']
    return files

def googleUpload(file_path, at_directory_path, startid='root'):
    name = file_path.split('/')[-1]
    if at_directory_path == '':
        dir_id = startid
    else:
        dir_id = googleGetIdOfPath(at_directory_path, startid=startid)
    mime = 'application/octet-stream'
    meta = {'name':name,'parents':[dir_id],'mimeType':mime}
    media = MediaFileUpload(file_path,mimetype=mime,resumable=True)
    file = service.files().create(body=meta,media_body=media,fields="id").execute()
    return file

def googleDownload(file_path, file_id): 
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request, chunksize=1048576)
    done = False
    # idx = 0
    while not done:
        # print(idx)
        # idx += 1
        _, done = downloader.next_chunk()
    fh.seek(0)
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(fh, f)

models = googleListFilesAtPath('MasterThesis/source/data/models', startid='1WM_oTurqZZTUg1GmfvE76USVXB1J_D0q')

for idx in range(len(models)):
    print('{} / {}: {}'.format(idx+1,len(models),models[idx]['name']))
    googleDownload('data/models_google/'+models[idx]['name'],models[idx]['id'])
    service.files().update(fileId=models[idx]['id'],body={'trashed': True}).execute()
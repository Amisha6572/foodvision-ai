import zipfile
zip_ref = zipfile.ZipFile('data/food11-image-dataset.zip', 'r')
zip_ref.extractall('data/food11')
zip_ref.close()
print('Done!')

import os
from subprocess import call

path_base = 'OriginalPDF/'
path_years = ['2015/', '2016/']
path_classes = ['category1/', 'category2/']

path_txt_base = 'dataset/'

for path_year in path_years:
    for path_class in path_classes:
        path_pdf_file = path_base + path_year + path_class
        for filename in os.listdir(path_pdf_file):
            path_pdf = path_pdf_file + filename
            path_txt = path_txt_base + path_year + path_class + filename[:-4] + '.txt'
            call(["python", "pdfminer.six-master/tools/pdf2txt.py", "-o", path_txt, path_pdf])

import sys

from pypdf import PdfReader

# 用于让 Claude 运行以确定 PDF 是否具有可填写的表单字段的脚本。参见 forms.md。


reader = PdfReader(sys.argv[1])
if reader.get_fields():
    print("This PDF has fillable form fields")
else:
    print("This PDF does not have fillable form fields; you will need to visually determine where to enter data")

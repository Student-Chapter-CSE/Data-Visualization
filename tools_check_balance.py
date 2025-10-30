import pathlib
p=pathlib.Path(r'c:\Users\abhra\Data-Visualization\dreamjournal.py')
s=p.read_text(encoding='utf-8').splitlines()
pb=0;qb=0
maxpb=(0,0);maxqb=(0,0)
for i,l in enumerate(s,1):
    pb+=l.count('(')-l.count(')')
    qb+=l.count('[')-l.count(']')
    if pb>maxpb[0]: maxpb=(pb,i)
    if qb>maxqb[0]: maxqb=(qb,i)
print('parens max',maxpb)
print('brackets max',maxqb)
print('\nline for parens max:')
print(maxpb[1], s[maxpb[1]-1])
print('\nline for brackets max:')
print(maxqb[1], s[maxqb[1]-1])
print('\ncontext parens:')
for ln in range(maxpb[1]-3,maxpb[1]+3):
    if 1<=ln<=len(s): print(ln, s[ln-1])
print('\ncontext brackets:')
for ln in range(maxqb[1]-3,maxqb[1]+3):
    if 1<=ln<=len(s): print(ln, s[ln-1])

import os
def findNremove(path,pattern,maxdepth=1):
    cpath=path.count(os.sep)
    for r,d,f in os.walk(path):
        if r.count(os.sep) - cpath <maxdepth:
            for files in f:
                if files.startswith(pattern):
                    try:
                        #print "Removing %s" % (os.path.join(r,files))
                        os.remove(os.path.join(r,files))
                    except Exception(e):
                        print(e)
                    else:
                        print("%s removed" % (os.path.join(r,files)))
if __name__ =='__main__':
	path = "./data"
	findNremove(path,"._",5)#"._"

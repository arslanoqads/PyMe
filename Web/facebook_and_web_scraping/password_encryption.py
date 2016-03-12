from passlib.hash import sha256_crypt as sha

x='pass'
p1=sha.encrypt(str(x))
p2=sha.encrypt('pass')

print p1
print p2


#verify the password with the prior value
print sha.verify('pass',p1)
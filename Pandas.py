import pandas as pd
s= pd.Series([9904312,3448737,2890451,2466052],index=["Seoul","Busan","Incheon","Daegue"])
#print(s)
#print(s.index)
#print(s.values)
#s.name="인구"
#s.index.name="도시"
#print(s.index.name)
#시리즈에 연산을 하면 value에만 적용된다
#print(s/100000)
#print(s[(250e4<s)&(s<500e4)])
#Pandas에서는 뒤에 나오는 숫자까지 포함하므로 주의해야한다.
#print(s[:3])
#s0=pd.Series(range(3),index=["a","b","c"])
#print(s0)
#print("서울" in s)
#for k,v in s.items():
#    print("%s=%d"%(k,v))
s2=pd.Series({"Seoul":9631482,"Busan":3393191,"Incheon":2632035,"Daejoen":1490158})
print(s2)
#딕셔너리의 원소는 순서를 가지지 않으므로 시리지의 데이터도 순서가 보장되지 않는다.
#만약 순서를 정하고 싶다면 인덱스를 리스트로 지정해야한다.
s2=pd.Series({"Seoul":9631482,"Busan":3393191,"Incheon":2632035,"Daejeon":1490158},
             index=["Busan","Seoul","Incheon","Daejeon"])
print(s2)
#인덱스 기반 연산
ds=s-s2
print(ds)
print(s.values-s2.values)
print(ds.notnull())
print(ds[ds.notnull()])

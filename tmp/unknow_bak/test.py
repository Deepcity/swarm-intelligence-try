class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        cran = list(ransomNote)
        cmag = list(magazine)
        ranmap = map(lambda x: (x,cran.count(x)), cran)
        magmap = map(lambda x: (x,cmag.count(x)), cmag)
        flag = True
        for (x,count) in ranmap:
            if count > magmap[x]:
                flag = False
                break
        if flag: print("true")
        else : print("false")
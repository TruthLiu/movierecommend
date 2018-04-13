from recommend.utils.datasets import load_movielens_movies
from recommend.utils.datasets import load_movielens_users



#
# movies=load_movielens_movies('ml-1m/movies.dat',separator="::")
# print(type(movies))
# dictMovies={}
# #get top ten movieId
# for i in movies[:11]:
#     print(i[0]+"=="+i[1])
#     dictMovies[int(i[0])-1]=str(i[1])
# for (k,v) in dictMovies.items():
#     print("movieId:%s  movieName=%s"%(k,v))
#
#
# users=load_movielens_users('ml-1m/users.dat')
# dictUsers={}
# for user in users[:1]:
#     userId=int(user[0])-1
#     sex=user[1]
#     age=user[2]
#     occupation=user[3]
# print(str(userId)+"--"+sex+"--"+age+"--"+occupation)

class userInfo:
    def userInfo(userage,occupation):
        dictage={1:"Under 18",18:"18-24",25:"25-34",35:"35-44",45:"45-49", 50:"50-55",56:"56+"}
        dictoccupation={20:  "writer",19:  "unemployed",18:  "tradesman/craftsman",17:  "technician/engineer",16:  "self-employed",
                    15: "scientist",14:  "sales/marketing",13:  "retired",12:  "programmer",11:  "lawyer",10:  "K-12 student",
                    9: "homemaker",8:  "farmer",7:  "executive/managerial",6:  "doctor/health care",5:  "customer service",
                    4: "college/grad student",3:  "clerical/admin",2:  "artist",1:  "academic/educator", 0:  "other or not define"}
        userage1=dictage.get(int(userage))
        occupation1=dictoccupation.get(int(occupation))

        return userage1,occupation1



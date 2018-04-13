
import numpy as np
from scipy._lib.six import xrange

from recommend.bpmf import BPMF
from recommend.utils.evaluation import RMSE
from recommend.utils.datasets import load_movielens_1m_ratings, load_movielens_movies, load_movielens_users

#load user ratings
from test import userInfo

ratings=load_movielens_1m_ratings('ml-1m/ratings.dat')
n_user=max(ratings[:,0])
n_item=max(ratings[:,1])
ratings[:,(0,1)]-=1 #shift ids by 1 to let user_id &movie_id start from 0

#fit model
bpmf=BPMF(n_user=n_user,n_item=n_item,n_feature=10,
          max_rating=5.,min_rating=1.,seed=0).fit(ratings,n_iters=5)

#traing RMSE
rmse=RMSE(bpmf.predict(ratings[:,:2]),ratings[:,2])
print("RMSE= 1 ---",rmse)

#predict rating for user 0 and item 0 to 9
#输入的用户id
userId=5
#输入要推荐的电影集合item 0 to endmovieNum-1
endmovieNum=n_item
#输入要显示的前五个movie
topN=5
array=bpmf.predict(np.array([[userId,i] for i in xrange(endmovieNum)]))
movies=load_movielens_movies('ml-1m/movies.dat')

dictMovies={}
for i in movies[:endmovieNum]:
    dictMovies[int(i[0])-1]=str(i[1])
# for (k,v) in dictMovies.items():
#     print("movieId:%s  movieName=%s"%(k,v))

users=load_movielens_users('ml-1m/users.dat')
user=users[userId]
age,occupation=userInfo.userInfo(user[2],user[3])
# userId=user[0]-1
sex=user[1]
print("userId:%s sex:%s age:%s occupation:%s"%(userId,sex,age,occupation))
count=0
movieidp={}
for item in array:
    movieidp[count]=item
    count=count+1
movieidps=sorted(movieidp.items(),key=lambda item:item[1],reverse=True)
for k,v in movieidps[:topN]:
    print("MovieName: %s porbability: %s"%(dictMovies.get(k),v))































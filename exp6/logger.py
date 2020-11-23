import visdom
import numpy as np



from multiprocessing import Process
import os
def openvis():
    os.system("python -m visdom.server ")

class logger():
    def __init__(self,strs="",str2="2333"):
        p1=Process(target=openvis)
        p1.start()
        self.vis = visdom.Visdom()
        self.vis.check_connection()
        self.legend=["loss"]

        if len(strs)>3:
            self.win=strs
        else:
            self.win = self.vis.line(Y=np.array([0, 0]), X=np.array([0, 0]),opts=dict(title=str2) ,name="loss")
        print(str(self.win))
        self.defx=0
        self.vis.update_window_opts(win=self.win, opts=dict(legend=self.legend))
        # vis.updateTrace(X=np.array([3]),Y=np.array([5]),win=self.win   ,name="loss")
        #
        # vis.line(X=np.array([0]), Y=np.array([0]), win=self.win, name="323", update="new")
        # self.legend.append("323")
        # vis.update_window_opts(win=self.win, opts=dict(legend=self.legend))
        # vis.updateTrace(X=np.array([1,10]), Y=np.array([9,10]), win=self.win,name="323")

    def refline(self,y=0,title="",x=-1):
        if x==-1:
            x=self.defx
        if title not in self.legend:
            print("ref")
            self.legend.append(title)
            print(self.legend)
            self.vis.line(X=np.array([x]), Y=np.array([y]), win=self.win, name=title, update="new")
            self.vis.update_window_opts(win=self.win, opts=dict(legend=self.legend))
            # vis.line(X=np.array([0]), Y=np.array([0]),win=self.win,update="new", name=line)
        self.vis.updateTrace(X=np.array([x]),Y=np.array([y]),win=self.win,name=title)

if __name__=="__main__":
    log=logger()
    log.refline(10, 7, "max")
    log.refline(5,6,"max")
    log.refline(7, 6, "1")

    # vis.line(
    #     X=np.array([0,1]),
    #     Y=np.array([1,2]),
    #     opts=dict(markercolor=np.array([50]),
    #              markersymbol='dot',),
    #     ##选择之前的窗口win
    #     win=win,
    #     ##选择更新图像的方式,另外有"append"/"new"
    #     update="new",
    #     ## 对当前的line新命名, 这个必须要
    #     name="2",)



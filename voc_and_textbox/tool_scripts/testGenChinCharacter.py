#coding=utf-8
import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np


DroidSansFallbackFull = ImageFont.truetype("/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/DroidSansFallbackFull.ttf", 36, 0)
fangzheng_fangsong = ImageFont.truetype("/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/fangzheng_fangsong.ttf", 36, 0)
fangzheng_heiti = ImageFont.truetype("/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/fangzheng_heiti.ttf", 36, 0)
fangzheng_shusong = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/fangzheng_shusong.ttf',36,0)
Lantinghei = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/Lantinghei.ttc',36,0)
mingliu = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/mingliu.ttc',36,0)
NotoSansHans_Black = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/NotoSansHans-Black.otf',36,0)
NotoSansHans_Bold = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/NotoSansHans-Bold.otf',36,0)
NotoSansHans_DemiLight = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/NotoSansHans-DemiLight.otf',36,0)
NotoSansHans_Light = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/NotoSansHans-Light.otf',36,0)
NotoSansHans_Medium = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/NotoSansHans-Medium.otf',36,0)
NotoSansHans_Regular = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/NotoSansHans-Regular.otf',36,0)
NotoSansHans_Thin_Windows = ImageFont.truetype('/home/wz/DeepLearning/OCR_working_dir/CNN_Chinese_ocr_training/font/NotoSansHans-Thin-Windows.otf',36,0)


fonts = [DroidSansFallbackFull,
         fangzheng_fangsong,
         fangzheng_heiti,
         fangzheng_shusong,
         Lantinghei,
         mingliu,
         NotoSansHans_Black,
         NotoSansHans_Bold,
         NotoSansHans_DemiLight,
         NotoSansHans_Light,
         NotoSansHans_Medium,
         NotoSansHans_Regular,
         NotoSansHans_Thin_Windows]




def affineTrans(image,mode,size=(48,48)):
    if mode == 0:# padding移动
        which = np.array([0,0,0,0])
        which[np.random.randint(0,4)]= np.random.randint(0,10)
        which[np.random.randint(0, 4)] = np.random.randint(0, 10)
        image = cv2.copyMakeBorder(image,which[0],which[0],which[0],which[0],cv2.BORDER_CONSTANT,value=0)
        image = cv2.resize(image,size)
    if mode == 1:
        scale = np.random.randint(48,int(48*1.4))
        center = [scale/2,scale/2]
        image = cv2.resize(image, (scale,scale))
        image = image[center[0]-24:center[0]+24,center[1]-24:center[1]+24]



    return image







def augmentation(image,mode):
    #np.random.random(size)

    if mode == 0:
        image = cv2.GaussianBlur(image,(5,5),np.random.randint(1,10))
    if mode == 1:
        image = cv2.GaussianBlur(image, (5, 5), np.random.randint(1,6))
        flag,image = cv2.threshold(image,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    if mode == 2:#横向断针
        for i in xrange(0,image.shape[0],2):
            cv2.line(image,(0,i),(48,i),0,1)

    if mode == 3:#纵向断针
        for i in xrange(0, image.shape[0], 2):
            cv2.line(image, (i, 0), (i, 48), 0, 1)
    if mode == 4:
        for i in xrange(0,image.shape[0],2):
            cv2.line(image,(0,i),(48,i),0,1)
        for i in xrange(0, image.shape[0], 2):
            cv2.line(image, (i, 0), (i, 48), 0, 1)


    if mode == 6:
        size = 5
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        image = cv2.filter2D(image, -1, kernel_motion_blur)

        #kernal = []
        #image =
    if mode == 7:
        size = 5
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        image = cv2.filter2D(image, -1, kernel_motion_blur)

    return image



def noise(image,mode=1):
    noise_image = (image.astype(float)/255)+(np.random.random((48, 48))*(np.random.random()*0.3))
    norm = (noise_image - noise_image.min())/(noise_image.max() - noise_image.min())
    if mode == 1:

        norm  =(norm * 255).astype(np.uint8)
    return norm





def DrawChinese(txt,font):
    image = np.zeros(shape=(48,48),dtype=np.uint8)
    x = Image.fromarray(image)
    draw = ImageDraw.Draw(x)
    draw.text((8,2),txt,(255),font=font)
    p = np.array(x)
    # p  = cv2.cvtColor(p,cv2.COLOR_RGB2BGR)
    return p




char_set = u"0123456789QWERTYUIOPASDFGHJKLZXCVBNM“”《》的一是不人有了在你我个大中要这为上生时会以就子到来可能和自们年多发心好用家出关长他成天对也小后下学都点国过地行信方得最说二业分作如看女于面注别经动公开现而美么还事己理维没之情高法全很日体里工微者实力做等水加定果去所新活着让起市身间码品进孩前想道种识按同车本然月机性与那无手爱样因老内部每更意号电其重化当只文入产合些她三费通但感常明给主名保提将元话气从教相平物场量资知或外度金正次期问放头位安比真务男第解原制区消路及色网花把打吃系回此应友选什表商再万妈被并两题服少风食变容员交儿质建民价养房门需影请利管白简司代口受图处才特报城单西完使已目收十候山数展快强式精结东师求接至海片清各直带程世向先任记持格总运联计觉何太线又免热件权调专医乐效神击设钱健流由见台几增病投易南导功介证走今光朋即视造您立改母推眼复政买传认非基宝营院四习越包游转技条息血科难规众喜便创干界示广红住欢源指该观读享深油达告具取轻康型周装张五满店亲标查育配字类优始整据考案北它客火必购办社命味步护术阅吧素户往菜适边却失节料较形近级准皮衣书马超照值父怎试空切找华供米企助反望香足福且排阳统未治决确项除低根岁则百备像早领酒款防集环富财跟致瘦速择温销团离呢议论吗王州态思参许远责布编随细春克听减言招组景穿黄药肉售股首限检修验共约段笑洗况续底园帮引婚份历济险士错语村伤局票善校战际益职够晚极支存旅故含算送诉留角松积省仅江境称半星升象材预群获青终害肤属显卡餐银声站队落假县饭补研连德哪钟遇黑双待毒断充智演讲压农愿尽拉粉响死牌古货玩苦率千施蛋器楼痛究睡状订义绝石亮势音搭委斯居李紧坚脸独依丽严止疗右喝鸡牛林板某负京丰句评融军懂吸划念夫层降哦税豆彩官络胸拿画尔龙察班构秘否叫球幸座慢兴佛室啊均付模协互置般英净换短左版课茶策毛停河肥答良久承控激范章云普套另奖须例写灵担志顾草镇退希谢爸采六鱼围密庭脑奇八卖童土圈谁拥糖监甚怕贵顺鲜冷差梦警拍铁亿争夜背永街律饮继刻初突倒聘木熟婆列频虽刚妆举尚汽曾脚奶破静驾块蓝酸核锅艺绿博额陈坐靠巧掉飞盘币腿巴培若闻史亚纸症季叶乡丝询剧礼七址添织略虚迎摄余乎缺胃爆域妻练荐临佳府追患树颜诚伴湖贴午困似测肝归宁暖纳宜阿异卫录液私谈泡惊索盐漂损稳休折讯堂怀惠汤纪散藏湿透令冰妇麻醒宣抗典执秀肌训刘急赶播苏淡革阴批盖腰肠脱印硬促冲床努脏跑雅厅罗惯族姐犯罪赛趣骨烧哈避征劳载润炒软慧驶妹占租馆累签副键煮尊予缘港雨兰斤呼申障坏竟疑顶饰九炎歌审戏借误辆端沙掌恶疾露括固移脂武寒零烟毕雪登朝聚笔姓波涨救厂央咨党延耳危斑汉沉夏侧鞋牙媒腹龄励瓜敢忙宽箱释操输抱野癌守搞染姜默翻哥洁娘挑凉末潮违附杀宫迷杂弱岛础贫析乱乳辣弃桃轮浪赏抽镜盛胜玉烦植绍恋冒缓渐虑肯赚绩忘珍恩针猪既聊蜜握舞甜败汇抓刺骗杯啦灯赞寻仍陪涉椒荣哭欲词巨圆刷概沟幼尤偏斗胡启尼述弟屋田判触柔忍架吉肾狗欧遍甘瓶综曲威齐桥纯阶贷丁伙眠罚逐韩封扎厚著督冬舒杨惜汁庆迪洋洲旧映疼席暴漫辈射鼓葱侵羊倍挂束幅碗裤胖旺川搜航弹嘴派脾届托库唯奥菌君途讨券距粗诗授祛谓序账凡晓峰剂筑敏肚暗辑访岗腐痘摩烈扬谷纹遗偿穷帝尿腾禁竞豪苹跳挥抢卷胆递珠敬甲乘孕绪纷隐滑浓膜姑探宗姻诺摆狂篇睛闲勇蒜尾旦庄窗扫辛陆塑幕聪详污圳扮肿楚忆匀炼耐衡措铺薪泰懒贝磨怨鼻圣孙眉泉洞焦毫戴旁符泪邮爷钢混厨抵灰献扣怪碎擦胎缩扶恐欣顿伟丈皇蒙胞尝寿攻仁津潜滴晨颗舍秒刀酱悲妙隔桌册迹仔闭奋袋墙嫌萝唐跌尖莫拌赔忽宿扩胶雷燕衰挺宋湾脉凭丹繁拒肺涂郁剩仪紫滋泽薄森唱残虎档猫麦劲偶秋疯俗悉弄船雄兵晒扰蒸悟肪览籍丑拼诊吴循偷灭伸赢魅勤旗亡乏估替吐碰淘彻逼氧梅遭孔稿嘉卜赵姿储呈乌娱闹裙倾震忧貌萨塞鬼池沿畅盟仙醋炸粥咖瑜返稍灾肩殊逃荷描朱朵横徐杰陷迟莱纠榜债烂伽拟匙圾巾恼誉垃颈壁链糊悦屏浮魔毁拜宾迅芝燃迫疫柜烤塔赠伪阻绕饱辅醉抑撒粘丢卧徒奔锁董枣截番蔬摇亦趋冠呀疲婴诸贸泥伦嫁祖朗琴拔孤摸壮帅阵梁宅啥伊鲁怒熊艾裁犹撑莲吹纤昨谱咳蜂闪嫩瞬霸兼恨昌踏瑞樱萌厕郑寺愈傻慈汗奉缴暂爽堆浙忌慎坦撞耗粒仿诱凤矿锻馨尘兄杭虫熬赖恰恒鸟猛唇幻窍浸诀填亏覆盆彼腺胀苗竹魂吵刑惑岸傲垂呵荒页抹揭贪宇泛劣臭呆梯啡径咱筹娃鉴禅召艳澳恢践迁废燥裂兔溪梨饼勺碍穴坛诈宏井仓删挣柳戒腔涵寸弯隆插祝氏泌盒邀煤膏棒跨拖葡骂喷肖洛盈浅逆夹贤晶厌侠欺敌钙冻盾桂仰滚萄厦牵疏齿挡孝滨吨渠囊慕捷淋桶脆沫辉耀渴邪轨悔猎煎沈虾醇贯衫荡谋携晋糕玻肃杜皆秦盗臂舌杆俱棉挤拨剪阔稀腻骑玛忠伯伍狠宠勒浴勿媳晕佩屈纵奈抬栏菲坑茄雾坡幽跃坊枝凝拳谨筋菇锋璃郭钻酷愁摘捐谐遵苍飘搅漏泄祥锦衬矛猜凌挖喊猴芳曼痕鼠允叔牢绘嘛吓振墨烫厉昆拓卵凯淀皱枪尺疆姆笋粮邻菩署柠遮艰芽爬夸捞叹缝妨奏岩寄吊狮剑驻洪夺募凶辨崇莓斜檬悬瘤欠刊曝傅悠椅戳棋慰丧拆绵炉徽驱曹履俄兑闷赋狼愉纽膝饿窝辞躺瓦逢堪薯哟袭壳咽岭槽雕昂闺御旋寨抛祸殖喂俩贡狐弥遥桑搬陌陶乃寂滩妥辰堵蛇侣邦蝙陵洒浆蹲惧霜丸娜扔肢姨援炫岳迈躁蝠埋泻巡溶氛械翠陕乔漠滞哲浩驰摊糟铜赤谅蕉昏劝杞扭骤杏娇渡抚羡娶串碧叉廉膀柱垫伏痒捕咸瓣庙敷卑碑沸鸭纱赴盲辽疮浦逛愤黎耍咬仲枸催喉瑰勾臀泼椎翼奢郎杠碳谎悄瓷绑酬菠朴割砖惨埃霍耶仇嗽塘邓漆蹈鹰披橘薇溃拾押炖霉痰袖巢帽宴卓踪屁刮晰豫玫驭羞讼茫厘扑亩鸣罐澡惩刹啤揉纲腥沾陀蟹枕逸橙梳浑毅吕泳碱缠柿砂羽黏芹馈柴侦卢憾疹贺牧俊峡硕倡蓄赌吞躲旨崩寞碌堡唤韭趁惹衷蛮译彭掩攀慌牲栋鼎鹅弘敲诞撕卦腌葛舟寓氨弗伞罩芒沃棚契巷磁浇逻廊棵溢箭匹矩颇爹玲绒雀鸿贩锐曰蕾竭剥沪畏渣歉摔旬颖茂擅铃淮叠挫逗晴柏舰翁框涌琳罢辩勃霞裹烹庸臣莉匆熙轩骄煲掘搓乙痴恭韵渗薏炭痣锡丨脊夕丘苑蔡裸灌庞龟窄兽敦辟牺僧怜湘篮妖喘瘾蓬挽芦谦踩辱辖捧坠滤炮撩狱亭虹吻煌谊枯脐螺扇抖戚怖帐盼冯劫墓崔酵殿蝶袁袜枚芯绳颠耕壶叨乖呕筷捡鹿潘笨扁渔株斥砸涩倦沥丛翔吼裕翰蒂尸莴暑肴凰馅阁誓匠侯韧钥哒狸媚壤驴逝渍嘲颁谜翅笼冈蓉脖甩扯宙叛帖萧芬潭涛闯泊宰梗鑫祭嚼卸尬尴怡咒晾嚣哄掏哀盯腊灿涯钞轰髦斌茅骚咋茨蝇枢捣顽彰拘坎役砍皂汪孟筱愚滥妒塌轿窃喻胁钓墅糙浏愧赫捏妮溜谣膳郊睫沧撤搏汰鹏菊帘秤衔捉鹤贿廷撼钾绽轴凸魄晃磷蒋栽荆蠢魏蜡缸筒遂茎芭伐邵瞎帕凑唠祈赁秩辫玄酶潇稻兜婷栓屡削钉拭蕴糯煞坪兹妃兆沂纺酿柚瀑稠腕勉疡贱冀跪凹辜铭赐绎灶弛嫉姚慨褐翘饶焯蒲哎僵隙犬剖昧湛矮舆吾剔甄逾虐粹牡莎罕蠕拐琪瑟霾辐帆拇榨冤绣痔筛雇祷歪贼肛垢抄饺琐裔黛睁捂萎酥饥衍靓榄嗯肆咯槛寡诵贬瞧乞贾弓珊眸屑熏籽乾聆狭韦锈毯蹄涤磊赘歇坝豹橄葬竖奴磅蝴淑柯敛侈叙惫俞翡叮蜀逊葩拯咪喔灸橱函厢瑶橡俯沛嘱佣陇莞妄榆淫靖俏敞嫂烘腑崖扒洽宵膨亨妞硫剁秉淤婉稣筝屌挨儒哑铅斩阱钩睿彬啪琼桩萍蔓焖踢铝仗荠棍棕铸榴惕巩杉芋攒髓拦蝎飙栗畜挪冥藤坤嘿磕椰憋荟坞屯饲懈梭夷嘘沐蔗蚕粕吁卉昭饪钮恳睦讶穆拣傍岂蘸噪戈靴瑕龈讽泣浊哇趾蔽丫歧蚊暨钠芪艇暮擎畔禽拧惟俭蔚恤蚀尹侍馒锌骼咏堕渊桐窒焕阀藕耻躯薛菱谭豁昕喧藉丙鸦驼拢奸爪睹绸暧佐颊澜禄缀煸趟揽蘑瘀阜拎屎颤邑胰肇哺噢矫讳雌怠楂苛暇酪佑妍婿耿妊萃灼丶澄撰弊挚庐雯靡牟硝酮醛苓紊肘趴廓卤昔鄂哮赣汕貅渝媛貔彦荫觅蹭巅岚甸漓迦邂稚濮陋逅窑笈弧颐禾瘙脓刃愣拴旭蚁滔仕荔琢澈睐隶粤盏遣汾镁硅枫淹仆胺娠舅弦殷惰麟苔芙堤旱蛙驳羯涕侨铲糜烯扛腮猿烛昵韶莹洱诠襄棠鸽仑峻啃瞒喇绊胱咙踝褶娩鲍掀漱绅奠芡蜗疤兮矣熔俺掰拱骏贞姥哼倘栖屉眷渭幢芜溺茯袍淳沦绞倪缚碟雁孵粪崛舱褪诡悍芸宪壹诫窟葵呐锤摧碾鞭嗓呱芥"

#
for one in char_set:
    image  = DrawChinese(one,fonts[np.random.randint(0,12)])
    image = noise(image)
    image = augmentation(image,np.random.randint(0, 8))
    image = noise(image)
    cv2.imshow("F", image)
    cv2.waitKey(0)



#from keras import
def norm(image):
    return image.astype(np.float)/255




# def GneranateBatch():



def Genernate(batchSize,charset):
#    pass
    label = []
    training_data = []

    for x in xrange(batchSize):
        char_id = np.random.randint(0, len(charset))
        font_id = np.random.randint(0,len(fonts))
        y = np.zeros(dtype=np.float,shape=(len(charset)))
        image = DrawChinese(charset[char_id],fonts[font_id])
        image = affineTrans(image,np.random.randint(0,2))
        image = noise(image)
        image = augmentation(image,np.random.randint(0,8))
        image = noise(image)
        image_norm = norm(image)
        image_norm  = np.expand_dims(image_norm,2)

        training_data.append(image_norm)
        y[char_id] = 1
        label.append(y)

    return np.array(training_data),np.array(label)


def Genernator(charset,batchSize):



    while(1):
        label = [];
        training_data = [];
        for i in xrange(batchSize):
            char_id = np.random.randint(0, len(charset))
            font_id = np.random.randint(0,len(fonts))
            y = np.zeros(dtype=np.float,shape=(len(charset)))
            image = DrawChinese(charset[char_id],fonts[font_id])
            image = affineTrans(image,np.random.randint(0,2))
            image = noise(image)
            image = augmentation(image,np.random.randint(0,8))
            image = noise(image)
            image_norm = norm(image)
            image_norm  = np.expand_dims(image_norm,2)
            y[char_id] = 1
            training_data.append(image_norm)
            label.append(y)




        y[char_id] = 1


        yield (np.array(training_data),np.array(label))

















from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


def Getmodel(nb_classes):
    # nb_classes = len(charset)
    nb_epoch = 30

    img_rows, img_cols = 48, 48
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 4

    # x = np.load('x.npy')
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=( img_rows, img_cols, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model





def eval(model,X,Y):
    res  = model.predict(X)

def Training(charset):
    model = Getmodel(len(charset))

    while(1):
        X, Y = Genernate(64, charset)
        model.train_on_batch(X,Y)
        print model.loss

        #Y = model.predict_classes(X)

def TrainingWithGenerator(charset,test  = 1):
    set = Genernate(1024, char_set)
    model = Getmodel(len(charset))
    BatchSize = 1024
    model.fit_generator(generator=Genernator(charset,BatchSize),samples_per_epoch=BatchSize*50,nb_epoch=60,validation_data=set)
    model.save("chi_sim.h5")

    X = set[0]
    Y = set[1]
    if test == 1:
        for i,one in enumerate(X):
            x = one

            res  = model.predict_classes(np.array([x]))[0]
            print u"预测结果",char_set[res],u"真实结果",char_set[Y[i].argmax()]
            image = (x.squeeze()*255).astype(np.uint8)
            cv2.imshow("sample",image)
            cv2.waitKey(0)


    model.save("./chi_sim.h5")


# BatchSize = 64
#
#TrainingWithGenerator(char_set)
#Training(char_set)




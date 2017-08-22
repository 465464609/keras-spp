import numpy as np #数学计算库，用于大型矩阵操作
from keras.layers import Input
from keras.models import Model
from RoiPoolingConv import RoiPoolingConv #可以直接调用文件放在同一个目录下
import keras.backend as K
import pdb
dim_ordering = K.image_dim_ordering()
assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

pooling_regions = 2
num_rois = 4
num_channels = 12

if dim_ordering == 'tf':
	in_img = Input(shape=(None, None, num_channels))
elif dim_ordering == 'th':
	in_img = Input(shape=(num_channels, None, None))

in_roi = Input(shape=(num_rois, 4))#in_roi是一个（4,4）的输入值

out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([in_img, in_roi])#第二个（）里的是Input的参数
#out_roi_pool 为输出，如果有下一层就作为他的输入
#中间变量可以共用一个，比如X=Dense(input);X=Dense(X);output=Dense(X)

model = Model([in_img, in_roi], out_roi_pool)
#一次性完成模型建立，第一个参数为输入，第二个为输出
model.summary()

model.compile(loss='mse', optimizer='sgd')

for img_size in [32]:
	if dim_ordering == 'th':
		X_img = np.random.rand(1, num_channels, img_size, img_size)
		#创建一个给定类型的数组，将其填充在一个均匀分布的随机样本[0, 1)中
		#返回的是随机数。类型为（）里描述的类型
		row_length = [float(X_img.shape[2]) / pooling_regions]
		col_length = [float(X_img.shape[3]) / pooling_regions]
	elif dim_ordering == 'tf':
		X_img = np.random.rand(1, img_size, img_size, num_channels)
		row_length = [float(X_img.shape[1]) / pooling_regions]
		col_length = [float(X_img.shape[2]) / pooling_regions]


	X_roi = np.array([[0, 0, img_size/2, img_size/2],
	                  [0, img_size/2, img_size/2, img_size/2],
	                  [img_size/2, 0, img_size/2, img_size/2],
	                  [img_size/2, img_size/2, img_size/2, img_size/2]])#一串放置ROI的数组，后期可以由selective search或者SLIC获得

	X_roi = np.reshape(X_roi, (1, num_rois, 4))#讲X_roi reshape 成一个（1，num_rois,4）的数组？？

	Y = model.predict([X_img, X_roi])

	for roi in range(num_rois):

		if dim_ordering == 'th':
			X_curr = X_img[0, :, X_roi[0, roi, 1]:X_roi[0, roi, 1]+X_roi[0, roi, 3], 
			X_roi[0, roi, 0]:X_roi[0, roi, 0]+X_roi[0, roi, 2]]
			row_length = float(X_curr.shape[1]) / pooling_regions
			col_length = float(X_curr.shape[2]) / pooling_regions
		elif dim_ordering == 'tf':
			X_curr = X_img[0, X_roi[0, roi, 1]:X_roi[0, roi, 1]+X_roi[0, roi, 3], 
			X_roi[0, roi, 0]:X_roi[0, roi, 0]+X_roi[0, roi, 2], :]
			row_length = float(X_curr.shape[0]) / pooling_regions
			col_length = float(X_curr.shape[1]) / pooling_regions

		idx = 0

		for ix in range(pooling_regions):
			for jy in range(pooling_regions):
				for cn in range(num_channels):

					x1 = int((ix * col_length))
					x2 = int((ix * col_length + col_length))
					y1 = int((jy * row_length))
					y2 = int((jy * row_length + row_length))
					dx = max(1,x2-x1)
					dy = max(1,y2-y1)
					x2 = x1+dx
					y2 = y1+dy

					if dim_ordering == 'th':
						m_val = np.max(X_curr[cn, y1:y2, x1:x2])
						if abs(m_val - Y[0, roi, cn, jy, ix]) > 0.01:
							pdb.set_trace()
						np.testing.assert_almost_equal(
							m_val, Y[0, roi, cn, jy, ix], decimal=6)
						idx += 1
					elif dim_ordering == 'tf':
						m_val = np.max(X_curr[y1:y2, x1:x2, cn])
						if abs(m_val - Y[0, roi, jy, ix, cn]) > 0.01:
							pdb.set_trace()
						np.testing.assert_almost_equal(
							m_val, Y[0, roi, jy, ix, cn], decimal=6)
						idx += 1

print('Passed roi pooling test')

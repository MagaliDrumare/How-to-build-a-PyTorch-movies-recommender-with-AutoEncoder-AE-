# What is an AutoEncoder? 
* Autoencoders - Ep. 10 (Deep Learning SIMPLIFIED): https://youtu.be/s96mYcicbpE
* Auto Encoder - definition by Hugo Larochelle : https://youtu.be/FzS3tMl4Nsc
* Auto Encoder - loss function by Hugo Larochelle : https://youtu.be/xTU79Zs4XKY
* Auto Encoder Deep Learning A-Z Udemy : http://bit.ly/2iJJ4kj

# How to make a movie recommender with a stack autoEncoder 
* Building an AutoENcoder, Deep Learning Udemy A-Z : http://bit.ly/2gPqG5k

```python
### Part 1 : Archirecture of the AutoEncoder 

#nn.Module is a parent class 
# SAE is a child class of the parent class nn.Module
class SAE(nn.Module): 
# self is the object of the SAE class 

# Archirecture 
	def __init__(self, ): 
	# self can use alll the methods of the class nn.Module
		super(SAE,self).__init__()
	# Full connected layer  n°1, input and 20 neurons-nodes of the first layer
	# one neuron can be the genre of the movie
	
	# Encode step 
		self.fc1 = nn.Linear(nb_movies,20)
	# Full connected layer n°2 
		self.fc2 = nn.Linear(20,10)
	
	# Decode step 
	# Full connected layer n°3
		self.fc3 = nn.Linear(10,20) 
	# Full connected layer n°4
		self.fc4 = nn.Linear(20,nb_movies) 
	# Sigmoid activation function 
		self.activation = nn.Sigmoid()

# Action : activation of the neurons
def forward(self, x) : 
		x = self.activation(self.fc1(x))
		x = self.activation(self.fc2(x))
		x = self.activation(self.fc3(x))
		# dont's use the activation function 
		# use the linear function only 
		x = self.fc4(x)
		# x is th evector of predicted ratings
		return x 

# Create the AutoEncoder object 
sae=SAE()
#MSE Loss : imported from torch.nn 
criterion=nn.MSELoss() 
# RMSProp optimizer (update the weights) imported from torch.optim 
#sea.parameters() are weights and bias adjusted during the training
optimizer=optim.RMSProp(sae.parameters(),lr=0.01, weight_decay=0.5)

### Part 2 : Training of the SAE 
# number of epochs 
nb_epochs = 200 
# Epoch forloop 
for epoch in range(1, nb_epoch+1): 
		# at the beginning the loss is at zero
		s=0.
		train_loss = 0 

		#Users forloop 
		for id_user in range(nb_users)
			# add one dimension to make a two dimension vector.
			# create a new dimension and put it the first position .unsqueeze[0]
			input = Variable(training_set[id_user].unsqueeze[0])
			
			# clone the input to obtain the target  
			target= input.clone()
			
			# target.data are all the ratings 
			# ratings > 0
			if torch.sum(target.data >0) > 0
				output = sae(input)
				# don't compute the gradients regarding the target
				target.require_grad=False 
				# only deal with true ratings 
				output[target==0]=0
				
				# Loss Criterion 
				loss =criterion(output,target)
				
				# Average of the error of the movies that don't have zero ratings
				mean_corrector=nb_movies/float(torch.sum(target.data>0)+1e-10)
				
				# Direction of the backpropagation 
				loss.backward()
				train_loss+=np.sqrt(loss.data[0]*mean_corrector)
				s+=1.
				
				# Intensity of the backpropagation 
				optimizer.step()
		
	print('epoch:' +str (epoch)+'loss:' +str(train_loss/s))
```


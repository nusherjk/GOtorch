package main

import (
	"fmt"
)


func calculatelossSE(yact, yhat float32) float32{
	return (yact-yhat)*(yact-yhat)
}

type output struct{
	value float32
	grad float32
	prevous_op []*output
	param_truth bool
	flag bool
	param_index int


}



type parameters struct{
	parameter_value float32
	gradient float32
}



type compgraph struct{
	
	parameter [] parameters
	param_index int
	
}

func (c *compgraph) add2x(input1, input2 output) output{
	out := output{
		value: input1.value + input2.value,
		grad: 1,
		param_truth : false,
		flag: true,

	}

	out.prevous_op = append(out.prevous_op, &input1)
	out.prevous_op = append(out.prevous_op, &input2)

	return out

}

func (c *compgraph) relu(input output) output{

	if (input.value>0){
		out := output{
		value: input.value,
		grad: 1,
		param_truth: false,
		flag: true,

	}
	out.prevous_op = append(out.prevous_op, &input)
	return out



	} else {
		out := output{
		value: 0,
		grad: 0,
		param_truth: false,
		flag: true,

	}
	out.prevous_op = append(out.prevous_op, &input)
	return out

	}
	
}
	
func (c *compgraph) mult(input output, index int) output{
	
	out := output{
		value : input.value* c.parameter[index].parameter_value,
		grad: c.parameter[index].parameter_value,
		
		param_truth: true,
		flag:true,
		param_index: index,

	}
	out.prevous_op = append(out.prevous_op, &input)

	c.parameter[index].gradient = input.value
	//fmt.Println(c.param_index)

	return out

}





func (c *compgraph) add(input output, index int) output{
	
	out := output{
		value : input.value + c.parameter[index].parameter_value,
		grad: 1,
		
		param_truth: true,
		flag:true,
		param_index : index,

	}
	
	out.prevous_op = append(out.prevous_op, &input)
	c.parameter[index].gradient = 1
	//fmt.Println(c.param_index)
	//fmt.Println("add")
	//c.param_index = c.param_index +1


	return out
}

func (c *compgraph) sub(input output, index int) output{
	out := output{
		value : input.value - c.parameter[index].parameter_value,
		grad: 1,
		
		param_truth: true,
		flag:true,
		param_index : index,

	}
	
	out.prevous_op = append(out.prevous_op, &input)

	c.parameter[index].gradient = -1
	//fmt.Println(c.param_index)
	//fmt.Println("add")
	//c.param_index = c.param_index +1


	return out	
}


func (c *compgraph) square(input output) output{
	
	out := output{
		value : input.value *input.value,
		grad: 2*input.value,
		//prevous_op : &input,
		param_truth: false,
		flag:true,

	}
	out.prevous_op = append(out.prevous_op, &input)
	//fmt.Println(c.param_index)
	//fmt.Println("sqr")
	return out

	}

func initialize_param(size_of_param int) [500] parameters{
	var para [500] parameters
	for a:=0; a<size_of_param ; a++ {
		para[a].parameter_value = 1.0
		para[a].gradient = 0.0
	}
	
	return para
} 


func (c *compgraph) backpropagate(node output, dt float32) {
	if (node.flag== true){
		

		if(node.param_truth){
		    
			c.parameter[node.param_index].gradient = c.parameter[node.param_index].gradient*  dt

			fmt.Println("nodes param Index = ", node.param_index, ": " ,c.parameter[node.param_index].gradient )
		
		
		}
		fmt.Println("dt* node.grad :", dt ,"*" , node.grad)

		dt = dt*node.grad
		
			//c.parameter[node.param_index].gradient
			
		
		for i :=0 ; i<len(node.prevous_op); i++ {
			c.backpropagate(*node.prevous_op[i], dt)
		}
		

	}
	return
}



func model(X,Y float32, param []parameters) (float32, []parameters){

	input := output{
		value: X,
		flag : false,
		
	}
	
	
	e := compgraph{
       
	param_index: len(param),
        parameter : param,
        
	
    }

    fmt.Println("parameters: ", e.parameter)
     var i1 output = e.square(input)

     var i2 output = e.mult(i1, 0)

     var i3 output = e.mult(input, 1)

     var i4 output = e.add2x(i2, i3)

     var i5 output = e.add(i4, 2)


    //var i1 output= e.mult(input, 0)
    
    //var i2 output= e.add(i1, 1)
    
    //var i5 output= e.square(i2)

    var loss output 
    loss.value = calculatelossSE(Y, i5.value)


    loss.grad = -2*(Y-i5.value)
	loss.prevous_op = append(loss.prevous_op, &i5)
	loss.param_truth =false
	loss.flag = true

	fmt.Println("loss :  " ,loss)
    fmt.Println("###################################################################################")
    e.backpropagate(loss,1)

    return loss.value, e.parameter
    

    



	
	}



func conv(X []float32,Y float32,karnel []parameters)(float32, []parameters){


	var input [3]output
	input[0] = output{
		value: X[0],
		flag : false,
		
	}

	input[1] = output{
		value: X[1],
		flag : false,
		
	}

	input[2] = output{
		value: X[2],
		flag : false,
		
	}


	
	
	e := compgraph{
       
	param_index: len(karnel),
        parameter : karnel,
        }

    var i1 = e.mult(input[0], 0)
    var i2 = e.mult(input[1], 1)
    var i3 = e.mult(input[2], 2)


    //sliding dot


    var i4 = e.add2x(i1,i2)
    var i5 = e.add2x(i4, i3)

    //Loss calc

     var loss output 
    loss.value = calculatelossSE(Y, i5.value)


    loss.grad = -2*(Y-i5.value)
	loss.prevous_op = append(loss.prevous_op, &i5)
	loss.param_truth =false
	loss.flag = true

	fmt.Println("loss :  " ,loss)
    fmt.Println("###################################################################################")
    e.backpropagate(loss,1)


    return loss.value, e.parameter


}

func main() {

	var karnel_size  int = 3

	var size int = 9 

	var outputsize int = 9-3+1


	var X = []float32 {1,2,3,4,5,6,7,8,9}

	var Y = []float32 {1,2,3,4,5,6,7}

	var mseloss float32 = 0


	loss := make([]float32, size)
	kernel := make([]parameters, karnel_size)

	kernel[0].parameter_value = 1
	kernel[1].parameter_value = 0
	kernel[2].parameter_value = -1

	var convinp []float32


	for i:=0; i<outputsize; i++ {
		if(i+3>=size){
			break
		}
		convinp = X[i:i+3]
		loss[i], kernel = conv(convinp,Y[i], kernel)

		mseloss += loss[i]


	}

	mseloss = mseloss/float32(size)
	fmt.Println("loss: ", mseloss)

	











}

/*
func main() {
	var learning_rate float32= 0.001
	var trainable_size = 3
	var size = 2
	var X = []float32 {2,5}
	var Y = []float32 {7,31}
	var mseloss float32 = 0

	loss := make([]float32, size)
	hyperparam := make([]parameters, trainable_size)
	



	fmt.Println(hyperparam)
	for m:=0; m<size; m++ {
		loss[m], hyperparam = model(X[m],Y[m],hyperparam)
		fmt.Println(hyperparam)
		mseloss +=loss[m]

		
		

	
	}
	mseloss = mseloss/float32(size)
	fmt.Println("loss: ", mseloss)
	if mseloss>1 {
		for f:=0;f<trainable_size;f++ {
			hyperparam[f].parameter_value = hyperparam[f].parameter_value - learning_rate* hyperparam[f].gradient

		}

	}

	
	
	fmt.Println("works!!")
}
*/
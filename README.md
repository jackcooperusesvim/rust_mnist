# Machine Learning from Scratch (with Rust)

This is a project I am working on to replicate some of the functionality of popular machine learning libraries (tensorflow, keras, pytorch, etc.), but using only Rust. I am doing this to better understand the inner workings of Machine Learning Algorithms so I can catch up with current research and someday contribute to it myself.

Because I am writing this project to learn more about the actual algorithms, I will be writing all of my code with only two things: My current knowledge on the subject, and papers from ArXiv (sourced through paperswithcode.com). This means that I will be doing approximately 95% of the Vector Calculus myself, using a notebook I keep by my bed. I find it somewhat relaxing anyway.

Here is a quick list of my to-do:

note: I am using the concept of a [Hill Chart from "Shape Up"](https://basecamp.com/shapeup/3.4-chapter-13) to represent my progress in each area, where 0 is the start of the hill, 1 is the top, and 2 is the end.

## Currently Working On:
 - Designing a Data Structure for the relationship between Optimizers and Backpropogatable Layers (.75 hill)

## Next:
 - Do the calculus for Weights (1.4 hill)
 - Write a Categorical Loss Function (.8 hill)
 - Write a structure for a Layer Composition (.6 hill)
 - Write a structure for a full Neural Network (.6 hill)
 - Research L1 and L2 Regularization (0.0 hill)

 - Calculus for CNN (0.2 hill)
 - Calculus for RNN (0.0 hill)
 - Calculus for Transformer (0.0 hill)

## "Cool Idea. Maybe Someday.":
 - Move math to Rayon for speed (1 hill)
 - Optimize math to enable further speed improvements (0.2 hill)
     - Move math to CUDA for SPEEEEEED (0.7 hill)

## Completed:
 - Build Various Layers and Activation Functions
 - Design a Backpropogatable Vector Function Trait
 - Calculate Weight adjustments

 - Do the calculus for:
    - ReLu
    - Softmax (This one was the hardest so far)
    - Weighted layer connections

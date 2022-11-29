module Main (main) where
import System.Environment
import Control.Monad.State
import Control.Monad.Random
-- import Lib

-- helper functions
dot :: Num a => [a] -> [a] -> a
dot _ [] = 0
dot [] _ = 0
dot (x:xs) (y:ys) = (x * y) + dot xs ys

linear :: Num b => [b] -> [b] -> [b] -> [b]
linear x weights bias = map (+(dot x weights)) bias

sigmoid :: Floating a => a -> a -- the activation function, returns a value between 0 and 1 from a weighted set of inputs
sigmoid x = 1.0 / (1.0 + exp(-x))

dSigmoid :: Floating a => a -> a -- during back propagation, we need the derivative of the activation function in calculating gradients
dSigmoid x = sigmoid x * (1 - sigmoid x)

absoluteUnaveragedSum :: Num a => [a] -> [a] -> a
absoluteUnaveragedSum _ [] = 0
absoluteUnaveragedSum [] _ = 0
absoluteUnaveragedSum (a:as) (e:es) = abs (a - e) + absoluteUnaveragedSum as es

meanAbsoluteError :: Fractional a => [a] -> [a] -> a
meanAbsoluteError actual experimental = absoluteUnaveragedSum actual experimental / fromIntegral(length actual)

sortOutput :: (Ord a1, Fractional a1, Num a2) => a1 -> a2 -- converts the inference output into either 0 or 1 from the raw output of infer
sortOutput x
    | x < 0.5 = 0
    | otherwise = 1

-- some operator truth tables; basic enough to include as matrices here, used for training the model
-- [x, y, x OPERATOR y]
and = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
or = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
xor = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
nor  = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]]

-- randomWeights :: IO [Double] 
-- randomWeights = do
--     g <- newStdGen
--     return $ [x | x<-randoms g :: [Double], x < 0.1]

randomWeight :: (RandomGen g) => Rand g Double
randomWeight = do
  -- When you need random numbers, just call the getRandom* functions
  r <- getRandomR (0.0, 0.1)
  return $ r

randomWeights :: (RandomGen g) => Rand g (Double, Double)
randomWeights = do
    r <- randomWeight
    return (r, r)

{--
    backward propagation updates our weights and biases by calculated loss
    our function will need to first find the partial derivative of loss with respect to the inference our network make
    from there, we find partial derivatives of loss with respect to each weight and bias
    lastly, we subtract from each weight and bias the product of themselves and the learning rate, this is the gradient descent
--}
backwardPropagate :: Floating a => [a] -> [a] -> [a] -> [a] -> [a] -> [a] -> [a]
backwardPropagate inputVector outputVector hidden_weights output_weights hidden_bias output_bias= [hidden_weights_gradient, output_weights_gradient, hidden_bias_gradient, output_bias_gradient, sample_error] where

    {--
    in our feedforward neural network, data is fed forward through our input layer -> hidden layer -> output layer
    the function below is meant to calculate outputs from the network by summing the dot product of inputs and weights with the bias
    unlike infer(), this method returns both 1st and 2nd layer outputs since back propagation will need both to determine how to shift weights
    note: this method will be used for making predictions as well; a2 will be the interpretable result from the output layer
    --}
    z1 = linear inputVector hidden_weights hidden_bias
    a1 = map sigmoid $ z1
    z2 = linear a1 output_weights output_bias
    a2 = map sigmoid $ z2
   
    -- before finding weight gradients, we need the partial derivatives of layer 1 and 2 with respect to loss, these form the foundation of subsequent calculations
    dz2 = map (\x -> x - inputVector !! 0) a2 -- predicted result - true result, activated output layer derivative
    dz1 = replicate 2 $ sigmoid (z1 !! 0) * dot output_weights dz2 -- activated first layer derivative

    -- calculate hidden weights gradient, dLoss/dW1
    hidden_weights_gradient = dot inputVector dz1 / 2.0 -- output vector length will always be 2.0

    -- calculate output weights gradient, dLoss/dW2
    output_weights_gradient = dot a1 dz2 / 2.0

    -- calculate hidden bias gradient, dLoss/db1
    hidden_bias_gradient = foldr (+) 0 dz1 / 2.0

    -- calculate output bias gradient, dLoss/db2
    output_bias_gradient = foldr (+) 0 dz2 / 2.0

    -- calculate sample error
    sample_error = meanAbsoluteError (replicate 2 $ outputVector !! 0) a2

-- tunes each parameters by subtracting them by the product of their respective gradient calculated by backward propagation and the learning rate
updateParameters :: Num p => p -> [p] -> [p] -> [p] -> [p] -> [p] -> [[p]]
updateParameters learningRate backPropagateList hiddenWeights outputWeights hiddenBias outputBias = [u_hidden_weights, u_output_weights, u_hidden_bias, u_output_bias] where
    u_hidden_weights = map (\x -> x - (learningRate * backPropagateList !! 0)) hiddenWeights
    u_output_weights = map (\x -> x - (learningRate * backPropagateList !! 1)) outputWeights
    u_hidden_bias = map (\x -> x - (learningRate * backPropagateList !! 2)) hiddenBias
    u_output_bias = map (\x -> x - (learningRate * backPropagateList !! 3)) outputBias

-- make batches with randomized entries
-- def batch operator batchSize


main :: IO ()
main = do
    args <- getArgs

    g <- newStdGen
    g' <- newStdGen

    {--
    the neural network will take two input vectors, so we need two hidden weights: one for each input.
    each hidden weight will be randomly initialized to meet the expectation of our optimization algorithm: stochastic gradient descent (SGD)
    randomly initialized weights also help with what is called "symmetry breaking"; when all weights are initialized equally, it can become difficult for them to change independently when training
    research into the topic suggests that small values between 0 and 0.1 make for the best starting weights for SGD
    --}
    let hidden_weights = [fst $ evalRand randomWeights g, snd $ evalRand randomWeights g]
    let output_weights = [fst $ evalRand randomWeights g', snd $ evalRand randomWeights g']
    let hidden_bias = replicate 2 0.0
    let output_bias = replicate 2 0.0

    putStrLn $ "Hidden weights: " ++ show hidden_weights
    putStrLn $ "Output weights: " ++ show output_weights
    putStrLn $ head args

    -- use forM monad for training monad

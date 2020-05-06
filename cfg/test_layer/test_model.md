# TPU Limit for (edgecompiler : 2.1.302470888)
## model_1 (MNIST)

    * INPUT_SIZE : [784]
    * OUTPUT_SIZE : [10]
    * INPUT_NODE_NAMES : ["inputs"]
    * OUTPUT_NODE_NAMES : ["outputs"]
    
    ```
      目前測試可以過
    ```

## model_2(limit input)

    * INPUT_SIZE : [784]
    * OUTPUT_SIZE : [10]
    * INPUT_NODE_NAMES : ["inputs"]
    * OUTPUT_NODE_NAMES : ["outputs"]
    
    ```
      目前測試 ：輸入最大限制為 640x640x3
      
      input -> conv2d_3x3_32 
    ```
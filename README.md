# TEWA-RL
TEWA-RL: Threat Evaluation &amp; Weapon Assignment with Reinforcement Learning

## **Overview**  
TEWA-RL is a **custom Gymnasium environment** designed for **Threat Evaluation and Weapon Assignment (TEWA)**. The project explores two different approaches to solving the TEWA problem:

1. **Full Reinforcement Learning (RL)** – The model learns both **threat evaluation and weapon assignment** through training.  
2. **Hybrid RL-Auction Approach** – **Threat evaluation is handled by RL**, while **weapon assignment is optimized using an auction algorithm** for better efficiency.  

The environment simulates a battlefield scenario where autonomous agents must evaluate incoming threats and allocate weapons accordingly.

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://gitlab.com/your-username/tewa-rl.git
cd tewa-rl
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Train or Run a Model**  
- **Train an RL model:**  
  ```bash
  python train_model.py
  ```
- **Run a trained model:**  
  ```bash
  python run_model.py
  ```

## **Training & Evaluation**  
The **full RL version** learns to evaluate threats and assign weapons from scratch, while the **auction-based version** offloads the assignment process to an optimization algorithm.  

- **Recurrent PPO** is used for learning to handle a variable number of threats.  
- **Rewards are structured** to encourage prioritizing high-risk threats, maintaining stable assignments, and eliminating threats efficiently.  
- **TensorBoard integration** helps track training progress.  

## **Customizing the Environment**  
Modify `TEWAenv.py` to:
- Change **battlefield size, number of threats, weapons, and missiles.**  
- Adjust **reward structure** for different behaviors.  
- Switch between **full RL and hybrid RL-Auction methods.**  
- Experiment with **different RL algorithms (PPO, A2C, DQN, etc.).**  





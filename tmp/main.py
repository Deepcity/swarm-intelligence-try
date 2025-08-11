import numpy as np
import matplotlib.pyplot as plt

class Swarm:
    """群智能算法基类。"""
    def __init__(self, objective_function, num_particles, num_iterations):
        self.objective_function = objective_function  # 目标函数
        self.num_particles = num_particles  # 粒子数量
        self.num_iterations = num_iterations  # 迭代次数
        self.best_solution = None  # 全局最优解
        self.best_fitness = float('inf')  # 全局最优适应度

    def optimize(self):
        """优化过程（需在子类中实现）。"""
        raise NotImplementedError

class ParticleSwarmOptimization(Swarm):
    """粒子群优化算法类。"""
    def __init__(self, objective_function, num_particles, num_iterations, dimensions, bounds):
        super().__init__(objective_function, num_particles, num_iterations)
        self.dimensions = dimensions  # 解的维度
        self.bounds = bounds  # 解的边界
        self.particles = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
        self.velocities = np.zeros_like(self.particles)  # 粒子速度
        self.personal_best = self.particles.copy()  # 每个粒子的历史最优位置
        self.personal_best_fitness = np.full(num_particles, float('inf'))  # 每个粒子的历史最优适应度
        
        # 记录轨迹
        self.trajectory = np.zeros((num_particles, num_iterations, dimensions))

    def optimize(self):
        """执行PSO优化并可视化过程。"""
        plt.ion()  # 开启交互模式
        fig, ax = plt.subplots()
        
        for iteration in range(self.num_iterations):
            # 计算适应度
            fitness = np.apply_along_axis(self.objective_function, 1, self.particles)

            # 更新个人和全局最优
            for i in range(self.num_particles):
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness[i]
                    self.personal_best[i] = self.particles[i]
                if fitness[i] < self.best_fitness:
                    self.best_fitness = fitness[i]
                    self.best_solution = self.particles[i]

            # 更新速度和位置
            inertia = 0.5
            cognitive_component = 1.5
            social_component = 1.5
            r1, r2 = np.random.rand(2)

            self.velocities = (inertia * self.velocities +
                               cognitive_component * r1 * (self.personal_best - self.particles) +
                               social_component * r2 * (self.best_solution - self.particles))
            self.particles += self.velocities

            # 边界处理
            self.particles = np.clip(self.particles, self.bounds[0], self.bounds[1])

            # 记录轨迹
            self.trajectory[:, iteration] = self.particles

            # 可视化粒子位置和全局最优解
            ax.clear()
            # 绘制轨迹
            for i in range(self.num_particles):
                ax.plot(self.trajectory[i, :iteration + 1, 0], 
                        self.trajectory[i, :iteration + 1, 1], 
                        c='lightgray', alpha=0.1)  # 设置较浅的颜色和透明度
            
            # 绘制当前粒子的位置
            ax.scatter(self.particles[:, 0], self.particles[:, 1], c='blue', label='Particles')
            # 绘制最佳解的位置
            ax.scatter(self.best_solution[0], self.best_solution[1], c='red', marker='x', s=100, label='Best Solution')
            ax.set_xlim(self.bounds[0], self.bounds[1])
            ax.set_ylim(self.bounds[0], self.bounds[1])
            ax.set_title(f"Iteration {iteration + 1}")
            ax.legend()
            plt.pause(0.5)  # 暂停0.5秒

        plt.ioff()  # 关闭交互模式
        plt.show()  # 显示最终结果

# Rastrigin 函数定义
def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# 实例化并运行PSO算法
pso = ParticleSwarmOptimization(rastrigin_function, num_particles=30, num_iterations=100, dimensions=2, bounds=(-30, 5.12))
pso.optimize()

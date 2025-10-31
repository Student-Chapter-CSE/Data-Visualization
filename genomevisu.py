import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import random

class DNAVisualizer:
    def __init__(self, sequence=None):
        """Initialize DNA visualizer with optional sequence"""
        if sequence is None:
            self.sequence = self.generate_random_sequence(100)
        else:
            self.sequence = sequence.upper()
        
        self.color_map = {'A': '#FF6B6B', 'T': '#4ECDC4', 'G': '#45B7D1', 'C': '#FFA07A'}
    
    def generate_random_sequence(self, length):
        """Generate a random DNA sequence"""
        bases = ['A', 'T', 'G', 'C']
        return ''.join(random.choice(bases) for _ in range(length))
    
    def validate_sequence(self, seq):
        """Validate if sequence contains only valid DNA bases"""
        valid_bases = set('ATGC')
        return all(base in valid_bases for base in seq)
    
    def plot_linear_sequence(self):
        """Visualize DNA sequence as a linear pattern"""
        fig, ax = plt.subplots(figsize=(14, 4))
        
        for i, base in enumerate(self.sequence):
            color = self.color_map[base]
            ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5))
        
        ax.set_xlim(0, len(self.sequence))
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title('DNA Sequence Linear Pattern', fontsize=14, fontweight='bold')
        ax.set_xlabel('Position', fontsize=12)
        ax.axis('off')
        
        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=self.color_map[base], label=base) 
                          for base in 'ATGC']
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.show()
    
    def plot_helix_pattern(self):
        """Visualize DNA as a double helix pattern"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        t = np.linspace(0, 4*np.pi, len(self.sequence))
        x1 = np.cos(t)
        y1 = np.sin(t)
        x2 = np.cos(t + np.pi)
        y2 = np.sin(t + np.pi)
        z = np.linspace(0, len(self.sequence), len(self.sequence))
        
        # Plot bases on helix
        for i, base in enumerate(self.sequence):
            color = self.color_map[base]
            # Strand 1
            ax.scatter(x1[i], z[i], color=color, s=100, edgecolors='black', linewidth=0.5, zorder=3)
            # Strand 2
            ax.scatter(x2[i], z[i], color=color, s=100, edgecolors='black', linewidth=0.5, zorder=3)
            
            # Connect strands with lines
            if i % 5 == 0:
                ax.plot([x1[i], x2[i]], [z[i], z[i]], 'gray', linewidth=1, alpha=0.5, zorder=1)
        
        # Draw helix backbone
        ax.plot(x1, z, 'k-', alpha=0.3, linewidth=2, zorder=0)
        ax.plot(x2, z, 'k-', alpha=0.3, linewidth=2, zorder=0)
        
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Sequence Position', fontsize=12)
        ax.set_title('DNA Double Helix Pattern', fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [plt.scatter([], [], color=self.color_map[base], s=100, label=base, edgecolors='black') 
                          for base in 'ATGC']
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_base_composition(self):
        """Plot nucleotide composition"""
        bases = ['A', 'T', 'G', 'C']
        counts = [self.sequence.count(base) for base in bases]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(bases, counts, color=[self.color_map[base] for base in bases], 
                      edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Nucleotide Base', fontsize=12)
        ax.set_title('DNA Base Composition', fontsize=14, fontweight='bold')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_gc_content_sliding_window(self, window_size=20):
        """Plot GC content using sliding window"""
        gc_content = []
        positions = []
        
        for i in range(len(self.sequence) - window_size):
            window = self.sequence[i:i+window_size]
            gc_count = window.count('G') + window.count('C')
            gc_percentage = (gc_count / window_size) * 100
            gc_content.append(gc_percentage)
            positions.append(i)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(positions, gc_content, linewidth=2, color='#45B7D1', marker='o', markersize=4)
        ax.fill_between(positions, gc_content, alpha=0.3, color='#45B7D1')
        
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('GC Content (%)', fontsize=12)
        ax.set_title(f'GC Content Distribution (Window: {window_size})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self):
        """Get and print sequence statistics"""
        print("\n=== DNA Sequence Statistics ===")
        print(f"Sequence Length: {len(self.sequence)}")
        print(f"Sequence: {self.sequence[:50]}{'...' if len(self.sequence) > 50 else ''}")
        print("\nNucleotide Counts:")
        for base in 'ATGC':
            count = self.sequence.count(base)
            percentage = (count / len(self.sequence)) * 100
            print(f"  {base}: {count} ({percentage:.2f}%)")
        
        gc_content = (self.sequence.count('G') + self.sequence.count('C')) / len(self.sequence) * 100
        print(f"\nGC Content: {gc_content:.2f}%")
        print("=" * 30 + "\n")


def main():
    # Example usage
    print("DNA Genome Pattern Visualization Tool")
    print("=" * 40)
    
    # Create visualizer with random sequence
    visualizer = DNAVisualizer()
    
    # Display statistics
    visualizer.get_statistics()
    
    # Create visualizations
    print("Generating visualizations...\n")
    
    visualizer.plot_linear_sequence()
    visualizer.plot_helix_pattern()
    visualizer.plot_base_composition()
    visualizer.plot_gc_content_sliding_window(window_size=20)
    
    # Example with custom sequence
    print("\nExample with custom sequence:")
    custom_seq = "ATGCATGCATGCATGCATGCATGC"
    custom_viz = DNAVisualizer(custom_seq)
    custom_viz.get_statistics()


if __name__ == "__main__":
    main()
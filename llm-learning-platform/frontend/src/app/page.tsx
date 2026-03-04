import Link from "next/link";
import {
  BookOpen,
  Brain,
  Sparkles,
  Layers,
  Zap,
  BarChart3,
  Heart,
  Settings2,
  Target,
  Gauge,
  Search,
  Server,
  MessageSquare,
  Shield,
  ScrollText,
} from "lucide-react";

const coreModules = [
  {
    id: "tokenization",
    title: "Tokenization Lab",
    description: "Explore BPE, WordPiece, and character tokenization algorithms interactively.",
    icon: BookOpen,
    color: "from-blue-500 to-cyan-500",
    difficulty: "Beginner",
    href: "/modules/tokenization",
  },
  {
    id: "embeddings",
    title: "Embedding Explorer",
    description: "Visualize word embeddings in 3D space with positional encodings.",
    icon: Sparkles,
    color: "from-purple-500 to-pink-500",
    difficulty: "Beginner",
    href: "/modules/embeddings",
  },
  {
    id: "attention",
    title: "Attention Visualizer",
    description: "Step through attention computation with interactive heatmaps.",
    icon: Brain,
    color: "from-amber-500 to-orange-500",
    difficulty: "Intermediate",
    href: "/modules/attention",
  },
  {
    id: "transformer",
    title: "Transformer Builder",
    description: "Assemble transformer blocks — choose norms, activations, and architectures.",
    icon: Layers,
    color: "from-emerald-500 to-teal-500",
    difficulty: "Intermediate",
    href: "/modules/transformer",
  },
  {
    id: "training",
    title: "Training Dashboard",
    description: "Train models end-to-end with real-time loss curves and hyperparameter tuning.",
    icon: Zap,
    color: "from-red-500 to-rose-500",
    difficulty: "Intermediate",
    href: "/modules/training",
  },
  {
    id: "inference",
    title: "Inference Playground",
    description: "Explore autoregressive generation with temperature, top-k, and top-p.",
    icon: BarChart3,
    color: "from-indigo-500 to-violet-500",
    difficulty: "Intermediate",
    href: "/modules/inference",
  },
];

const advancedModules = [
  {
    id: "rlhf",
    title: "RLHF & Alignment Lab",
    description: "Train reward models and run PPO/DPO alignment experiments.",
    icon: Heart,
    color: "from-rose-500 to-pink-500",
    difficulty: "Advanced",
    href: "/modules/rlhf",
  },
  {
    id: "lora",
    title: "LoRA & QLoRA Studio",
    description: "Apply low-rank adaptation and quantized fine-tuning techniques.",
    icon: Settings2,
    color: "from-cyan-500 to-blue-500",
    difficulty: "Advanced",
    href: "/modules/lora",
  },
  {
    id: "evaluation",
    title: "Evaluation Suite",
    description: "Compute BLEU, ROUGE, perplexity and run model benchmark comparisons.",
    icon: Target,
    color: "from-green-500 to-emerald-500",
    difficulty: "Advanced",
    href: "/modules/evaluation",
  },
];

const frontierModules = [
  {
    id: "inference-opt",
    title: "Inference Optimization",
    description: "KV caching, weight quantization, and speculative decoding techniques.",
    icon: Gauge,
    color: "from-orange-500 to-amber-500",
    difficulty: "Expert",
    href: "/modules/inference-opt",
  },
  {
    id: "interpretability",
    title: "Interpretability Lab",
    description: "Logit lens, activation patching, neuron analysis, and circuit tracing.",
    icon: Search,
    color: "from-teal-500 to-cyan-500",
    difficulty: "Expert",
    href: "/modules/interpretability",
  },
  {
    id: "distributed",
    title: "Distributed Training",
    description: "Data, model, and pipeline parallelism with ZeRO optimizer stages.",
    icon: Server,
    color: "from-violet-500 to-purple-500",
    difficulty: "Expert",
    href: "/modules/distributed",
  },
  {
    id: "prompt-eng",
    title: "Prompt Engineering",
    description: "Templates, analysis, and comparison of prompt engineering techniques.",
    icon: MessageSquare,
    color: "from-sky-500 to-blue-500",
    difficulty: "Intermediate",
    href: "/modules/prompt-eng",
  },
  {
    id: "safety",
    title: "AI Safety Center",
    description: "Safety evaluation, red-team attacks, and constitutional AI principles.",
    icon: Shield,
    color: "from-red-600 to-orange-500",
    difficulty: "Expert",
    href: "/modules/safety",
  },
  {
    id: "long-context",
    title: "Long Context Explorer",
    description: "RoPE scaling, ALiBi, and positional encoding methods for long contexts.",
    icon: ScrollText,
    color: "from-lime-500 to-green-500",
    difficulty: "Expert",
    href: "/modules/long-context",
  },
];

export default function HomePage() {
  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Hero Section */}
      <section className="mb-16 text-center">
        <h1 className="text-5xl font-bold mb-4">
          <span className="gradient-text">Master LLMs</span>
          <br />
          From First Principles
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
          Interactive visualizations, hands-on experiments, and guided learning
          paths to understand every component of modern Large Language Models.
        </p>
        <div className="flex gap-4 justify-center">
          <Link
            href="/modules/tokenization"
            className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors"
          >
            Start Learning
          </Link>
          <Link
            href="/dashboard"
            className="px-6 py-3 glass rounded-xl font-medium hover:bg-white/90 dark:hover:bg-gray-800/90 transition-colors"
          >
            View Dashboard
          </Link>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="glass rounded-2xl p-6 mb-12 grid grid-cols-4 gap-6 text-center">
        <div>
          <div className="text-3xl font-bold gradient-text">15+</div>
          <div className="text-sm text-muted-foreground">Interactive Modules</div>
        </div>
        <div>
          <div className="text-3xl font-bold gradient-text">40+</div>
          <div className="text-sm text-muted-foreground">Visualizations</div>
        </div>
        <div>
          <div className="text-3xl font-bold gradient-text">100%</div>
          <div className="text-sm text-muted-foreground">From Scratch</div>
        </div>
        <div>
          <div className="text-3xl font-bold gradient-text">0</div>
          <div className="text-sm text-muted-foreground">Dependencies on PyTorch</div>
        </div>
      </section>

      {/* Core Module Grid */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Core Modules</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {coreModules.map((mod) => (
            <Link key={mod.id} href={mod.href} className="glass rounded-2xl p-6 hover:scale-[1.02] transition-transform group">
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${mod.color} flex items-center justify-center mb-4`}>
                <mod.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold mb-2 group-hover:text-primary-500 transition-colors">{mod.title}</h3>
              <p className="text-sm text-muted-foreground mb-3">{mod.description}</p>
              <span className="text-xs px-2 py-1 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400">{mod.difficulty}</span>
            </Link>
          ))}
        </div>
      </section>

      {/* Advanced Module Grid */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Advanced Modules</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {advancedModules.map((mod) => (
            <Link key={mod.id} href={mod.href} className="glass rounded-2xl p-6 hover:scale-[1.02] transition-transform group">
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${mod.color} flex items-center justify-center mb-4`}>
                <mod.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold mb-2 group-hover:text-primary-500 transition-colors">{mod.title}</h3>
              <p className="text-sm text-muted-foreground mb-3">{mod.description}</p>
              <span className="text-xs px-2 py-1 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400">{mod.difficulty}</span>
            </Link>
          ))}
        </div>
      </section>

      {/* Frontier Module Grid */}
      <section>
        <h2 className="text-2xl font-bold mb-6">Frontier Research</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {frontierModules.map((mod) => (
            <Link key={mod.id} href={mod.href} className="glass rounded-2xl p-6 hover:scale-[1.02] transition-transform group">
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${mod.color} flex items-center justify-center mb-4`}>
                <mod.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold mb-2 group-hover:text-primary-500 transition-colors">{mod.title}</h3>
              <p className="text-sm text-muted-foreground mb-3">{mod.description}</p>
              <span className="text-xs px-2 py-1 rounded-full bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400">{mod.difficulty}</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import { Github, Linkedin, Mail, Phone, MapPin, ExternalLink, TrendingUp, Award, Zap, Users, BarChart3, Brain, Activity, Clock, DollarSign, Target, ChevronDown, Menu, X, FileText } from 'lucide-react';

const QuantPortfolio = () => {
  const [scrollY, setScrollY] = useState(0);
  const [menuOpen, setMenuOpen] = useState(false);
  const [typedText, setTypedText] = useState('');
  
  const titles = [
    'Quantitative Researcher',
    'AI/ML Specialist',
    'Senior Software Engineer',
    'Financial Engineer'
  ];
  const [currentTitleIndex, setCurrentTitleIndex] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    let index = 0;
    const currentTitle = titles[currentTitleIndex];
    const interval = setInterval(() => {
      if (index <= currentTitle.length) {
        setTypedText(currentTitle.slice(0, index));
        index++;
      } else {
        clearInterval(interval);
        setTimeout(() => {
          setCurrentTitleIndex((prev) => (prev + 1) % titles.length);
        }, 2000);
      }
    }, 100);
    return () => clearInterval(interval);
  }, [currentTitleIndex]);

  const liveMetrics = [
    { 
      icon: TrendingUp, 
      label: 'LSTM HFT', 
      value: '23.7%',
      subtext: '5μs Latency',
      color: 'text-green-400',
      gradient: 'from-green-500 to-emerald-500'
    },
    { 
      icon: DollarSign, 
      label: 'Live Trading', 
      value: '48.6%',
      subtext: 'Annual Return',
      color: 'text-yellow-400',
      gradient: 'from-yellow-500 to-orange-500'
    },
    { 
      icon: Target, 
      label: 'Ensemble Alpha', 
      value: '18.2%',
      subtext: '2.1 Sharpe',
      color: 'text-blue-400',
      gradient: 'from-blue-500 to-cyan-500'
    },
    { 
      icon: Users, 
      label: 'Healthcare AI', 
      value: '200K+',
      subtext: 'Users Served',
      color: 'text-purple-400',
      gradient: 'from-purple-500 to-pink-500'
    }
  ];

  const professionalExp = [
    {
      title: 'Senior Quantitative Finance Engineer',
      company: 'Bidias Capital Consulting LLC',
      period: '2024 - Present',
      achievements: [
        '28.4% annual returns with 1.89 Sharpe ratio (live trading)',
        '5μs inference time with 94.2% accuracy for real-time analysis',
        '11+ production systems with multi-strategy backtesting',
        'VQE/QAOA quantum algorithms for portfolio optimization'
      ],
      gradient: 'from-yellow-500 to-orange-500'
    },
    {
      title: 'Data Science Analyst - Tech Solutions',
      company: 'Verizon',
      period: 'June 2022 - February 2025',
      achievements: [
        'ML pipelines serving 10M+ customers daily',
        '25% reduction in system latency',
        '99.9% uptime across 15+ business applications',
        'Collaboration with 20+ engineering teams'
      ],
      gradient: 'from-red-500 to-pink-500'
    },
    {
      title: 'Healthcare Technology & AI Engineer',
      company: 'Texas Health & Human Services',
      period: 'February 2020 - June 2022',
      achievements: [
        '200K+ patient records analyzed daily (85% accuracy)',
        '$2M+ annual savings through predictive models',
        'Real-time alerts across 300+ healthcare providers',
        '20% reduction in patient churn'
      ],
      gradient: 'from-green-500 to-teal-500'
    },
    {
      title: 'Data Analyst - Product Development',
      company: 'Apple Inc.',
      period: 'September 2014 - December 2019',
      achievements: [
        'Predictive models for $50B+ annual revenue products',
        'ETL pipelines processing data from 40+ countries',
        'Technical initiatives serving millions of customers',
        'Advanced BI infrastructure and dashboards'
      ],
      gradient: 'from-gray-600 to-gray-800'
    }
  ];

  const repositoryProjects = [
    {
      title: '01-Deep-Learning-Finance',
      category: 'Deep Learning',
      description: 'Advanced deep learning models for financial markets including LSTM networks, attention mechanisms, and transformer architectures for price prediction and risk assessment.',
      metrics: ['95.2% Accuracy', '0.03 MSE', '2.1 Sharpe Ratio'],
      tech: ['TensorFlow', 'PyTorch', 'Keras', 'Pandas', 'NumPy'],
      gradient: 'from-blue-500 to-cyan-500'
    },
    {
      title: '02-Healthcare-Analytics',
      category: 'Healthcare AI',
      description: 'Comprehensive healthcare analytics platform with predictive modeling, clinical decision support, and population health management serving 200K+ users.',
      metrics: ['200K+ Users', '85% Accuracy', '$2M+ Savings'],
      tech: ['Scikit-learn', 'XGBoost', 'Streamlit', 'PostgreSQL', 'FastAPI'],
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      title: '03-Quantitative-Finance',
      category: 'Quantitative Finance',
      description: 'Advanced quantitative finance models including portfolio optimization, risk management, derivatives pricing, and algorithmic trading strategies.',
      metrics: ['18.2% Returns', '2.1 Sharpe Ratio', '8.3% Max DD'],
      tech: ['Python', 'QuantLib', 'NumPy', 'SciPy', 'Matplotlib'],
      gradient: 'from-yellow-500 to-orange-500'
    },
    {
      title: '04-Machine-Learning',
      category: 'Machine Learning',
      description: 'Production-ready machine learning systems with ensemble methods, feature engineering, and automated model selection for financial applications.',
      metrics: ['92% Accuracy', '0.95 AUC-ROC', '15% Lift'],
      tech: ['Scikit-learn', 'XGBoost', 'LightGBM', 'Optuna', 'SHAP'],
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      title: '05-Statistical-Analysis',
      category: 'Statistics',
      description: 'Advanced statistical analysis including time series analysis, hypothesis testing, Bayesian inference, and econometric modeling.',
      metrics: ['0.001 p-value', '99% Confidence', 'R² = 0.87'],
      tech: ['Python', 'Statsmodels', 'PyMC3', 'R', 'SPSS'],
      gradient: 'from-indigo-500 to-purple-500'
    },
    {
      title: '06-Visualizations-Results',
      category: 'Data Visualization',
      description: 'Interactive data visualizations and comprehensive results presentation with advanced charting, dashboards, and real-time monitoring.',
      metrics: ['50+ Charts', 'Real-time Updates', '99.9% Uptime'],
      tech: ['Plotly', 'Dash', 'D3.js', 'Bokeh', 'Streamlit'],
      gradient: 'from-cyan-500 to-blue-500'
    },
    {
      title: '07-Research-Papers',
      category: 'Research',
      description: 'Academic research papers and publications on quantitative finance, machine learning applications, and financial engineering innovations.',
      metrics: ['5+ Publications', '50+ Citations', 'H-index: 3'],
      tech: ['LaTeX', 'R', 'MATLAB', 'Python', 'Stata'],
      gradient: 'from-rose-500 to-pink-500'
    },
    {
      title: '08-Advanced-ML-Finance',
      category: 'Advanced ML',
      description: 'Cutting-edge machine learning applications in finance including reinforcement learning for trading, GANs for synthetic data, and graph neural networks.',
      metrics: ['28.4% RL Returns', '95% GAN Quality', '0.92 Graph Accuracy'],
      tech: ['TensorFlow', 'PyTorch', 'Stable-Baselines3', 'DGL', 'Ray'],
      gradient: 'from-violet-500 to-purple-500'
    },
    {
      title: '09-High-Performance-Trading',
      category: 'High-Frequency Trading',
      description: 'Ultra-low latency trading systems with microsecond execution, FPGA acceleration, and advanced market microstructure analysis.',
      metrics: ['5μs Latency', '23.7% Returns', '99.99% Uptime'],
      tech: ['C++', 'CUDA', 'FPGA', 'Boost', 'ZeroMQ'],
      gradient: 'from-emerald-500 to-teal-500'
    },
    {
      title: '10-Performance-Results',
      category: 'Performance Analytics',
      description: 'Comprehensive performance analysis with detailed metrics, backtesting results, risk attribution, and comparative analysis.',
      metrics: ['48.6% Annual Return', '2.84 Sharpe Ratio', '89% Win Rate'],
      tech: ['Python', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn'],
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      title: '11-Live-Trading-Systems',
      category: 'Live Trading',
      description: 'Production trading systems with verified 48.6% annual returns, real-time execution, and comprehensive risk management.',
      metrics: ['48.6% Live Returns', '2.84 Sharpe Ratio', '$12.3M Profits'],
      tech: ['Python', 'OANDA API', 'Qiskit', 'Redis', 'PostgreSQL'],
      gradient: 'from-yellow-500 to-orange-500'
    }
  ];

  const skills = [
    { category: 'Quantitative Finance', items: 'Portfolio Optimization, Risk Modeling, Algorithmic Trading, Derivatives Pricing, VaR, Monte Carlo', level: 95, color: 'from-yellow-400 to-orange-400' },
    { category: 'AI/ML & Deep Learning', items: 'TensorFlow, PyTorch, LSTM, CNN, Transformers, XGBoost, Neural Networks', level: 93, color: 'from-purple-400 to-pink-400' },
    { category: 'Python & Programming', items: 'NumPy, Pandas, Scikit-learn, R, SQL, JavaScript, MATLAB', level: 95, color: 'from-blue-400 to-cyan-400' },
    { category: 'Healthcare Technology', items: 'Clinical ML, Predictive Modeling, Risk Scoring, Medical Imaging, Population Health', level: 88, color: 'from-green-400 to-emerald-400' },
    { category: 'Cloud & DevOps', items: 'AWS SageMaker, Docker, Kubernetes, CI/CD, GitHub Actions', level: 85, color: 'from-indigo-400 to-purple-400' },
    { category: 'Financial Data APIs', items: 'Bloomberg Terminal, OANDA, Alpha Vantage, Quandl, Real-time Feeds', level: 90, color: 'from-cyan-400 to-blue-400' }
  ];

  return (
    <div className="bg-gray-950 text-white min-h-screen overflow-x-hidden">
      {/* Animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-48 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 -right-48 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl animate-pulse" style={{animationDelay: '2s'}}></div>
      </div>

      {/* Navigation */}
      <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        scrollY > 50 ? 'bg-gray-900/95 backdrop-blur-lg shadow-lg shadow-blue-500/10' : 'bg-transparent'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold shadow-lg shadow-blue-500/50">
                JB
              </div>
              <span className="text-white font-bold text-xl hidden sm:block">Joseph Bidias</span>
            </div>
            
            <div className="hidden md:flex space-x-8">
              {['Home', 'Performance', 'Experience', 'Projects', 'Skills', 'Contact'].map((item) => (
                <a
                  key={item}
                  href={`#${item.toLowerCase()}`}
                  className="text-gray-300 hover:text-cyan-400 transition-colors relative group"
                >
                  {item}
                  <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-gradient-to-r from-blue-500 to-cyan-500 group-hover:w-full transition-all duration-300"></span>
                </a>
              ))}
            </div>
            
            <button onClick={() => setMenuOpen(!menuOpen)} className="md:hidden text-white">
              {menuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
        
        {menuOpen && (
          <div className="md:hidden bg-gray-900/98 backdrop-blur-lg border-t border-blue-500/20">
            <div className="px-4 py-4 space-y-3">
              {['Home', 'Performance', 'Experience', 'Projects', 'Skills', 'Contact'].map((item) => (
                <a
                  key={item}
                  href={`#${item.toLowerCase()}`}
                  onClick={() => setMenuOpen(false)}
                  className="block text-gray-300 hover:text-cyan-400 transition-colors py-2"
                >
                  {item}
                </a>
              ))}
            </div>
          </div>
        )}
      </nav>

      {/* Hero Section */}
      <section id="home" className="relative min-h-screen flex items-center justify-center px-4">
        <div className="text-center z-10 max-w-6xl mx-auto">
          <div className="mb-6">
            <h1 className="text-5xl sm:text-7xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent animate-pulse">
              Joseph Bidias
            </h1>
            <div className="h-12 flex items-center justify-center">
              <h2 className="text-xl sm:text-3xl text-cyan-400 font-light">
                {typedText}<span className="animate-pulse">|</span>
              </h2>
            </div>
          </div>
          
          <p className="text-gray-400 text-lg sm:text-xl max-w-4xl mx-auto mb-8">
            Elite Quantitative Researcher with 7+ years building production ML systems • 28.4% live trading returns • 
            200K+ users served • MS Financial Engineering 2025
          </p>
          
          <div className="flex flex-wrap justify-center gap-4 mb-12">
            <a href="#performance" className="px-8 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg font-semibold hover:shadow-lg hover:shadow-cyan-500/50 transition-all transform hover:scale-105">
              View Performance
            </a>
            <a href="#projects" className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg font-semibold hover:shadow-lg hover:shadow-purple-500/50 transition-all transform hover:scale-105">
              Repository Projects
            </a>
            <a href="#contact" className="px-8 py-3 border-2 border-blue-500 rounded-lg font-semibold hover:bg-blue-500/10 transition-all">
              Contact Me
            </a>
          </div>
          
          <div className="flex justify-center gap-6">
            <a href="https://github.com/eaglepython" target="_blank" rel="noopener noreferrer" 
               className="text-gray-400 hover:text-cyan-400 transition-colors transform hover:scale-110">
              <Github size={28} />
            </a>
            <a href="https://linkedin.com/in/joseph-bidias-eaglepython" target="_blank" rel="noopener noreferrer"
               className="text-gray-400 hover:text-cyan-400 transition-colors transform hover:scale-110">
              <Linkedin size={28} />
            </a>
            <a href="mailto:bidias_consulting@outlook.com"
               className="text-gray-400 hover:text-cyan-400 transition-colors transform hover:scale-110">
              <Mail size={28} />
            </a>
          </div>
        </div>
        
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <ChevronDown size={32} className="text-cyan-400" />
        </div>
      </section>

      {/* Live Performance Metrics */}
      <section id="performance" className="py-20 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl sm:text-5xl font-bold text-center mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Elite Performance Metrics
          </h2>
          <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-cyan-500 mx-auto mb-12"></div>
          
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {liveMetrics.map((metric, index) => (
              <div key={index} className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20 hover:border-cyan-500/50 transition-all hover:shadow-lg hover:shadow-cyan-500/20 group">
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${metric.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <metric.icon className="text-white" size={24} />
                </div>
                <div className={`text-3xl font-bold mb-1 ${metric.color}`}>{metric.value}</div>
                <div className="text-sm text-gray-400">{metric.label}</div>
                <div className="text-xs text-gray-500 mt-1">{metric.subtext}</div>
              </div>
            ))}
          </div>

          {/* Additional Impact Metrics */}
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20">
              <h3 className="text-xl font-bold text-cyan-400 mb-3">Trading Performance</h3>
              <div className="space-y-2 text-sm text-gray-300">
                <div className="flex justify-between"><span>Live Returns:</span><span className="text-green-400 font-bold">48.6%</span></div>
                <div className="flex justify-between"><span>Sharpe Ratio:</span><span className="text-blue-400 font-bold">2.84</span></div>
                <div className="flex justify-between"><span>Win Rate:</span><span className="text-purple-400 font-bold">89%</span></div>
                <div className="flex justify-between"><span>Max Drawdown:</span><span className="text-yellow-400 font-bold">-8.3%</span></div>
              </div>
            </div>

            <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20">
              <h3 className="text-xl font-bold text-cyan-400 mb-3">Enterprise Impact</h3>
              <div className="space-y-2 text-sm text-gray-300">
                <div className="flex justify-between"><span>Users Served:</span><span className="text-green-400 font-bold">10M+</span></div>
                <div className="flex justify-between"><span>Revenue Impact:</span><span className="text-blue-400 font-bold">$50B+</span></div>
                <div className="flex justify-between"><span>Cost Savings:</span><span className="text-purple-400 font-bold">$2M+</span></div>
                <div className="flex justify-between"><span>System Uptime:</span><span className="text-yellow-400 font-bold">99.9%</span></div>
              </div>
            </div>

            <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20">
              <h3 className="text-xl font-bold text-cyan-400 mb-3">Technical Excellence</h3>
              <div className="space-y-2 text-sm text-gray-300">
                <div className="flex justify-between"><span>HFT Latency:</span><span className="text-green-400 font-bold">5μs</span></div>
                <div className="flex justify-between"><span>ML Accuracy:</span><span className="text-blue-400 font-bold">95%</span></div>
                <div className="flex justify-between"><span>Projects Built:</span><span className="text-purple-400 font-bold">25+</span></div>
                <div className="flex justify-between"><span>Experience:</span><span className="text-yellow-400 font-bold">7+ yrs</span></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Professional Experience */}
      <section id="experience" className="py-20 relative bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl sm:text-5xl font-bold text-center mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Professional Experience
          </h2>
          <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-cyan-500 mx-auto mb-12"></div>

          <div className="space-y-8">
            {professionalExp.map((exp, index) => (
              <div key={index} className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-8 border border-blue-500/20 hover:border-cyan-500/50 transition-all">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
                  <div>
                    <h3 className="text-2xl font-bold text-white mb-1">{exp.title}</h3>
                    <div className={`text-lg bg-gradient-to-r ${exp.gradient} bg-clip-text text-transparent font-semibold`}>
                      {exp.company}
                    </div>
                  </div>
                  <div className="text-gray-400 text-sm mt-2 md:mt-0">{exp.period}</div>
                </div>
                <ul className="space-y-2">
                  {exp.achievements.map((achievement, i) => (
                    <li key={i} className="flex items-start gap-2 text-gray-300">
                      <span className="text-cyan-400 mt-1">▹</span>
                      <span>{achievement}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Repository Projects */}
      <section id="projects" className="py-20 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl sm:text-5xl font-bold text-center mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Portfolio Repository Projects
          </h2>
          <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-cyan-500 mx-auto mb-12"></div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {repositoryProjects.map((project, index) => (
              <div key={index} className="group bg-gray-900/50 backdrop-blur-sm rounded-xl overflow-hidden border border-blue-500/20 hover:border-cyan-500/50 transition-all hover:shadow-2xl hover:shadow-cyan-500/20 hover:scale-105">
                <div className={`h-2 bg-gradient-to-r ${project.gradient}`}></div>
                
                <div className="p-6">
                  <div className="flex items-start justify-between mb-3">
                    <span className="px-3 py-1 bg-blue-500/20 text-blue-400 text-xs font-semibold rounded-full">
                      {project.category}
                    </span>
                    <div className="flex gap-2">
                      <a href="https://github.com/eaglepython/QUANT_AI_ML_PORTOFOLIO" target="_blank" rel="noopener noreferrer"
                         className="text-gray-400 hover:text-cyan-400 transition-colors">
                        <Github size={18} />
                      </a>
                      <a href="#projects" className="text-gray-400 hover:text-cyan-400 transition-colors">
                        <ExternalLink size={18} />
                      </a>
                    </div>
                  </div>
                  
                  <h3 className="text-xl font-bold mb-3 text-white group-hover:text-cyan-400 transition-colors">
                    {project.title}
                  </h3>
                  
                  <p className="text-gray-400 text-sm mb-4 line-clamp-3">
                    {project.description}
                  </p>
                  
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.tech.map((tech, i) => (
                      <span key={i} className="px-2 py-1 bg-gray-800/50 text-cyan-400 text-xs rounded">
                        {tech}
                      </span>
                    ))}
                  </div>
                  
                  <div className="border-t border-gray-800 pt-4 space-y-1">
                    {project.metrics.map((metric, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs text-gray-400">
                        <div className={`w-1.5 h-1.5 rounded-full bg-gradient-to-r ${project.gradient}`}></div>
                        <span>{metric}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section id="skills" className="py-20 relative bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl sm:text-5xl font-bold text-center mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Technical Expertise
          </h2>
          <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-cyan-500 mx-auto mb-12"></div>
          
          <div className="max-w-4xl mx-auto space-y-6">
            {skills.map((skill, index) => (
              <div key={index} className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20 hover:border-cyan-500/50 transition-all">
                <div className="flex justify-between mb-3">
                  <div>
                    <span className="text-white font-semibold text-lg">{skill.category}</span>
                    <div className="text-gray-400 text-sm mt-1">{skill.items}</div>
                  </div>
                  <span className="text-cyan-400 font-semibold text-lg">{skill.level}%</span>
                </div>
                <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                  <div 
                    className={`h-full bg-gradient-to-r ${skill.color} rounded-full transition-all duration-1000 shadow-lg`}
                    style={{width: `${skill.level}%`, boxShadow: `0 0 20px rgba(34, 211, 238, 0.5)`}}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20 relative">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl sm:text-5xl font-bold text-center mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Let's Connect
          </h2>
          <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-cyan-500 mx-auto mb-12"></div>
          
          <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-8 border border-blue-500/20">
            <p className="text-gray-300 text-center mb-8 text-lg">
              Available for senior quantitative research positions, AI/ML engineering roles, and consulting opportunities
            </p>
            
            <div className="grid sm:grid-cols-2 gap-4 mb-8">
              <a href="mailto:bidias_consulting@outlook.com" className="flex items-center gap-4 p-4 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-all group">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center flex-shrink-0 group-hover:shadow-lg group-hover:shadow-cyan-500/50 transition-all">
                  <Mail className="text-white" size={24} />
                </div>
                <div className="min-w-0">
                  <div className="text-sm text-gray-400">Primary Email</div>
                  <div className="text-white font-semibold truncate">bidias_consulting@outlook.com</div>
                </div>
              </a>
              
              <a href="mailto:rodabeck777@gmail.com" className="flex items-center gap-4 p-4 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-all group">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center flex-shrink-0 group-hover:shadow-lg group-hover:shadow-cyan-500/50 transition-all">
                  <Mail className="text-white" size={24} />
                </div>
                <div className="min-w-0">
                  <div className="text-sm text-gray-400">Alternate Email</div>
                  <div className="text-white font-semibold truncate">rodabeck777@gmail.com</div>
                </div>
              </a>
              
              <a href="tel:+12148863785" className="flex items-center gap-4 p-4 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-all group">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0 group-hover:shadow-lg group-hover:shadow-purple-500/50 transition-all">
                  <Phone className="text-white" size={24} />
                </div>
                <div className="min-w-0">
                  <div className="text-sm text-gray-400">Phone</div>
                  <div className="text-white font-semibold">(214) 886-3785</div>
                </div>
              </a>
              
              <a href="https://github.com/eaglepython" target="_blank" rel="noopener noreferrer" className="flex items-center gap-4 p-4 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-all group">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center flex-shrink-0 group-hover:shadow-lg group-hover:shadow-gray-500/50 transition-all">
                  <Github className="text-white" size={24} />
                </div>
                <div className="min-w-0">
                  <div className="text-sm text-gray-400">GitHub</div>
                  <div className="text-white font-semibold truncate">github.com/eaglepython</div>
                </div>
              </a>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a href="https://eaglepython.github.io/Software-Engineer-Portofolio/" target="_blank" rel="noopener noreferrer"
                 className="inline-flex items-center justify-center gap-2 px-8 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg font-semibold hover:shadow-lg hover:shadow-cyan-500/50 transition-all transform hover:scale-105">
                <FileText size={20} />
                Software Engineering Portfolio
              </a>
              
              <a href="https://bidiascapitalconsulting.netlify.app/" target="_blank" rel="noopener noreferrer"
                 className="inline-flex items-center justify-center gap-2 px-8 py-3 bg-gradient-to-r from-yellow-600 to-orange-600 rounded-lg font-semibold hover:shadow-lg hover:shadow-orange-500/50 transition-all transform hover:scale-105">
                <TrendingUp size={20} />
                Live Trading Platform
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-gray-400 text-sm">
              © 2025 Joseph Bidias. Elite Quantitative Researcher & AI/ML Specialist.
            </div>
            <div className="flex gap-6">
              <a href="https://github.com/eaglepython" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-cyan-400 transition-colors">
                <Github size={20} />
              </a>
              <a href="https://linkedin.com/in/joseph-bidias-eaglepython" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-cyan-400 transition-colors">
                <Linkedin size={20} />
              </a>
              <a href="mailto:bidias_consulting@outlook.com" className="text-gray-400 hover:text-cyan-400 transition-colors">
                <Mail size={20} />
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default QuantPortfolio;
import Image from "next/image";

export default function Home() {
  const metrics = [
    { label: "Total Patients", value: "500", color: "text-blue-600" },
    { label: "Heart Disease Rate", value: "55.0%", color: "text-red-600" },
    { label: "Average Age", value: "53.0 yrs", color: "text-green-600" },
    { label: "Avg Cholesterol", value: "349.1 mg/dl", color: "text-orange-600" },
  ];

  const findings = [
    "Focus screening on patients over 50 years old.",
    "Monitor cholesterol levels regularly for high-risk groups.",
    "Implement lifestyle interventions for patients with multiple risk factors.",
  ];

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Navigation / Header Bar */}
      <nav className="bg-white border-b border-gray-200 py-4 px-8 mb-8">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <span className="text-xl font-bold text-blue-600">RGT Healthcare Analytics</span>
          <div className="space-x-6 text-sm font-medium text-gray-500">
            <span>Dashboard</span>
            <span>Reports</span>
            <span>Settings</span>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-8">
        <header className="mb-12">
          <div className="inline-block bg-blue-100 text-blue-700 px-3 py-1 rounded-full text-xs font-semibold tracking-wider uppercase mb-4">
            Milestone Project A
          </div>
          <h1 className="text-4xl font-extrabold text-gray-900 mb-2 tracking-tight">Heart Disease Business Insights</h1>
          <p className="text-lg text-gray-600 max-w-2xl">
            A comprehensive analysis of patient data to identify key risk factors and provide actionable healthcare recommendations.
          </p>
        </header>

        <main>
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {metrics.map((metric, index) => (
              <div key={index} className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100 hover:shadow-md transition-shadow">
                <p className="text-sm font-medium text-gray-500 mb-1">{metric.label}</p>
                <p className={`text-3xl font-bold ${metric.color}`}>{metric.value}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
            {/* Overview Section */}
            <div className="lg:col-span-1 space-y-8">
              <section className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
                <h2 className="text-lg font-bold text-gray-900 mb-4">Project Overview</h2>
                <p className="text-sm text-gray-600 leading-relaxed">
                  This project transforms raw healthcare data from the UCI Heart Disease dataset into a business-ready insights pack.
                  The goal is to support medical decision-making through data-driven evidence.
                </p>
              </section>

              <section className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
                <h2 className="text-lg font-bold text-gray-900 mb-4">Methodology</h2>
                <ul className="text-sm text-gray-600 space-y-2">
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
                    Data Cleaning & Imputation
                  </li>
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
                    Exploratory Data Analysis
                  </li>
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
                    SQL Risk Factor Analysis
                  </li>
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
                    Visual Storytelling
                  </li>
                </ul>
              </section>
            </div>

            {/* Visualizations Area */}
            <div className="lg:col-span-2 space-y-8">
              <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
                <h2 className="text-xl font-bold text-gray-800 mb-6">Disease Distribution</h2>
                <div className="relative h-80 w-full">
                  <Image 
                    src="/disease_distribution.png" 
                    alt="Disease Distribution" 
                    fill 
                    className="object-contain"
                  />
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
                <h2 className="text-xl font-bold text-gray-800 mb-6">Age Analysis</h2>
                <div className="relative h-80 w-full">
                  <Image 
                    src="/age_by_disease.png" 
                    alt="Age by Disease" 
                    fill 
                    className="object-contain"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Findings & Recommendations */}
          <div className="bg-white rounded-2xl shadow-sm p-8 border border-gray-100 border-l-4 border-l-blue-600">
            <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
              <svg className="w-6 h-6 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              Key Findings & Clinical Recommendations
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {findings.map((finding, index) => (
                <div key={index} className="space-y-2">
                  <span className="text-blue-600 font-bold text-lg">0{index + 1}</span>
                  <p className="text-gray-700 leading-relaxed font-medium">{finding}</p>
                </div>
              ))}
            </div>
          </div>
        </main>

        <footer className="mt-20 pt-8 border-t border-gray-200 flex flex-col md:flex-row justify-between items-center text-gray-400 text-sm">
          <p>© 2026 RGT 2025 NSP AI/Data Training Program</p>
          <div className="flex space-x-4 mt-4 md:mt-0">
            <span className="hover:text-blue-500 cursor-pointer transition-colors">Documentation</span>
            <span className="hover:text-blue-500 cursor-pointer transition-colors">GitHub</span>
            <span className="hover:text-blue-500 cursor-pointer transition-colors">Support</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

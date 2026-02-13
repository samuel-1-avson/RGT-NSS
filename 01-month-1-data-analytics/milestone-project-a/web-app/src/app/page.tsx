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
    <div className="min-h-screen bg-gray-50 p-8">
      <header className="mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Heart Disease Business Insights</h1>
        <p className="text-lg text-gray-600">Milestone Project A: Healthcare Analytics Dashboard</p>
      </header>

      <main>
        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {metrics.map((metric, index) => (
            <div key={index} className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
              <p className="text-sm font-medium text-gray-500 mb-1">{metric.label}</p>
              <p className={`text-3xl font-bold ${metric.color}`}>{metric.value}</p>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Chart 1 */}
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Disease Distribution</h2>
            <div className="relative h-64 w-full">
              <Image 
                src="/disease_distribution.png" 
                alt="Disease Distribution" 
                fill 
                className="object-contain"
              />
            </div>
            <p className="mt-4 text-sm text-gray-500 italic text-center">
              Breakdown of heart disease prevalence across the patient sample.
            </p>
          </div>

          {/* Chart 2 */}
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Age by Disease Prevalence</h2>
            <div className="relative h-64 w-full">
              <Image 
                src="/age_by_disease.png" 
                alt="Age by Disease" 
                fill 
                className="object-contain"
              />
            </div>
            <p className="mt-4 text-sm text-gray-500 italic text-center">
              Visualization of how disease rate correlates with patient age.
            </p>
          </div>
        </div>

        {/* Findings & Recommendations */}
        <div className="bg-white rounded-xl shadow-sm p-8 border border-gray-100">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Key Recommendations</h2>
          <ul className="space-y-4">
            {findings.map((finding, index) => (
              <li key={index} className="flex items-start">
                <span className="flex-shrink-0 h-6 w-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center mr-3 font-bold">
                  {index + 1}
                </span>
                <p className="text-lg text-gray-700">{finding}</p>
              </li>
            ))}
          </ul>
        </div>
      </main>

      <footer className="mt-16 pt-8 border-t border-gray-200 text-center text-gray-500">
        <p>© 2026 RGT 2025 NSP AI/Data Training Program - Milestone Project A</p>
      </footer>
    </div>
  );
}

"use client";

import React, { useState, useEffect } from "react";

// ------------------------------------------------------------
// MODEL PARAMETERS (Extracted from scikit-learn training)
// ------------------------------------------------------------
const MEANS = [
  11526880.65484375, // pendapatan_per_b
  0.645070781642855,  // rasio_utang_pend (DTI capped at 2.0)
  2.0308125,          // jumlah_tunggakan
  37.71775,           // usia
  664.1559375         // skor_kredit_sebe
];

const STDS = [
  65568426.96369116,  // pendapatan_per_b
  0.5301913947846489,  // rasio_utang_pend
  2.566293648404981,   // jumlah_tunggakan
  9.580146655323185,   // usia
  78.69627728486331    // skor_kredit_sebe
];

const COEF = [
  -16.29570516789081,   // num__pendapatan_per_b
  0.13409386156314765,  // num__rasio_utang_pend
  0.7526754174810938,   // num__jumlah_tunggakan
  0.6173427641965604,   // num__usia
  0.08013682547295675,  // num__skor_kredit_sebe
  -1.4870878565225525,  // cat__jenis_pekerjaan_PNS
  -0.77065389802496,    // cat__jenis_pekerjaan_Pengusaha
  4.388988866248744,    // cat__jenis_pekerjaan_Tidak Bekerja
  0.9926082054409225    // cat__jenis_pekerjaan_Wiraswasta
];

const INTERCEPT = -1.9621160078836577;

// Pure JS inference function
function predictDefaultRisk(inputs) {
  // Hard business rule: minimum income is 2,500,000 IDR
  if (inputs.pendapatan < 2500000) {
    return 1.0;
  }

  // Clip DTI ratio to 2.0 to match the training pipeline clipping
  const cappedRasio = Math.min(inputs.rasio, 2.0);

  // Standardize numeric features
  const z_pend = (inputs.pendapatan - MEANS[0]) / STDS[0];
  const z_rasio = (cappedRasio - MEANS[1]) / STDS[1];
  const z_tunggakan = (inputs.tunggakan - MEANS[2]) / STDS[2];
  const z_usia = (inputs.usia - MEANS[3]) / STDS[3];
  const z_skor = (inputs.skor - MEANS[4]) / STDS[4];

  // One-Hot Encode (Reference category: "Karyawan Swasta")
  const is_pns = inputs.pekerjaan === "PNS" ? 1 : 0;
  const is_pengusaha = inputs.pekerjaan === "Pengusaha" ? 1 : 0;
  const is_tidak_bekerja = inputs.pekerjaan === "Tidak Bekerja" ? 1 : 0;
  const is_wiraswasta = inputs.pekerjaan === "Wiraswasta" ? 1 : 0;

  // Calculate logit z
  const z = INTERCEPT
    + (COEF[0] * z_pend)
    + (COEF[1] * z_rasio)
    + (COEF[2] * z_tunggakan)
    + (COEF[3] * z_usia)
    + (COEF[4] * z_skor)
    + (COEF[5] * is_pns)
    + (COEF[6] * is_pengusaha)
    + (COEF[7] * is_tidak_bekerja)
    + (COEF[8] * is_wiraswasta);

  // Return probability via sigmoid
  let prob = 1 / (1 + Math.exp(-z));

  // Hard business rule for DTI (Debt-to-Income) ratio
  if (inputs.rasio > 0.50) {
    prob = Math.max(prob, 0.85);
  } else if (inputs.rasio > 0.40) {
    prob = Math.max(prob, 0.65);
  }

  return prob;
}

// ------------------------------------------------------------
// METODOLOGI DATA CONFIG FOR FLOWCHART DOCUMENTATION
// ------------------------------------------------------------
const METODOLOGI_DATA = {
  strategies: {
    conservative: {
      name: "Konservatif",
      desc: "Fokus utama meminimalisir NPL (Non-Performing Loan). Menggunakan model dengan tingkat Recall tinggi untuk mengidentifikasi potensi gagal bayar sejak dini.",
      recall: "92% - 95%",
      precision: "70% - 75%",
      cutoffDesc: "Threshold cut-off rendah (0.30 - 0.45) sehingga sistem lebih sensitif dalam memblokir/menolak pengajuan kredit yang meragukan."
    },
    expansion: {
      name: "Ekspansi",
      desc: "Fokus utama meningkatkan volume penyaluran kredit dan market share. Menggunakan model yang seimbang untuk menyaring debitur dengan lebih adil tanpa memperketat batasan.",
      recall: "80% - 84%",
      precision: "85% - 88%",
      cutoffDesc: "Threshold cut-off lebih tinggi/longgar (0.50 - 0.65) untuk mempermudah persetujuan debitur potensial."
    }
  },
  creditTypes: {
    consumer: {
      name: "Consumer (Konsumer)",
      desc: "Kredit perorangan untuk kebutuhan konsumtif dengan agunan bervariasi. Risiko bervariasi berdasarkan kepemilikan aset."
    },
    micro: {
      name: "Mikro (Micro)",
      desc: "Kredit usaha ultra-mikro atau mikro dengan plafon kecil. Sangat bergantung pada arus kas usaha bulanan debitur."
    },
    small: {
      name: "Small (Ritel/Kecil)",
      desc: "Kredit modal kerja dan investasi untuk badan usaha kecil/menengah dengan dokumen keuangan formal."
    }
  },
  products: {
    kpr: {
      name: "KPR (Kredit Pemilikan Rumah)",
      type: "consumer",
      desc: "Kredit pemilikan hunian beragunan properti dengan tenor panjang (10-25 tahun).",
      thresholds: { conservative: 0.35, expansion: 0.55 },
      model: "Logistic Regression (Balanced) + Hard Asset Collateral Rules"
    },
    kkb: {
      name: "KKB (Kredit Kendaraan Bermotor)",
      type: "consumer",
      desc: "Kredit pemilikan kendaraan beragunan objek kendaraan dengan depresiasi aset berkala.",
      thresholds: { conservative: 0.40, expansion: 0.60 },
      model: "Logistic Regression (Standardized) + Vehicle Value Depreciation Filter"
    },
    kartu_kredit: {
      name: "Kartu Kredit",
      type: "consumer",
      desc: "Fasilitas kredit konsumsi jangka pendek menggunakan kartu dengan limit transaksional tertentu.",
      thresholds: { conservative: 0.28, expansion: 0.48 },
      model: "Logistic Regression + Card Scoring Engine"
    },
    ceria: {
      name: "Ceria (Digital Paylater)",
      type: "consumer",
      desc: "Pinjaman digital paylater untuk transaksi belanja online cepat tanpa agunan.",
      thresholds: { conservative: 0.32, expansion: 0.52 },
      model: "Logistic Regression + Digital Footprint Scoring"
    },
    briguna: {
      name: "Briguna (Kredit Tanpa Agunan)",
      type: "consumer",
      desc: "Kredit tanpa agunan (KTA) untuk pegawai aktif/pensiunan dengan sumber pembayaran payroll.",
      thresholds: { conservative: 0.30, expansion: 0.50 },
      model: "Logistic Regression + Payroll Verification Rules"
    },
    kur: {
      name: "KUR (Kredit Usaha Rakyat)",
      type: "micro",
      desc: "Kredit bersubsidi pemerintah untuk permodalan UMKM produktif.",
      thresholds: { conservative: 0.45, expansion: 0.65 },
      model: "Logistic Regression + Government Subsidy Risk Sharing Rule"
    },
    kur_kecil: {
      name: "KUR Kecil",
      type: "micro",
      desc: "Kredit Usaha Rakyat dengan plafond menengah/tertinggi untuk modal kerja dan investasi mikro.",
      thresholds: { conservative: 0.40, expansion: 0.60 },
      model: "Logistic Regression + Micro Scale Feasibility Rules"
    },
    kupedes: {
      name: "Kupedes (Kredit Umum Pedesaan)",
      type: "micro",
      desc: "Kredit umum non-subsidi untuk sektor pertanian, perdagangan, dan jasa pedesaan.",
      thresholds: { conservative: 0.38, expansion: 0.58 },
      model: "Logistic Regression + Cashflow Capacity Verification Model"
    },
    kmk: {
      name: "Kredit Modal Kerja Retail",
      type: "small",
      desc: "Kredit jangka pendek untuk pembiayaan modal kerja operasional perusahaan retail.",
      thresholds: { conservative: 0.35, expansion: 0.55 },
      model: "Logistic Regression + Receivable & Inventory Valuation Model"
    },
    investasi: {
      name: "Kredit Investasi Retail",
      type: "small",
      desc: "Kredit jangka menengah/panjang untuk pengadaan barang modal, ekspansi usaha, atau rehabilitasi pabrik.",
      thresholds: { conservative: 0.32, expansion: 0.52 },
      model: "Logistic Regression + Feasibility Study & IRR Constraints Model"
    }
  }
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  // Navigation State
  const [currentTab, setCurrentTab] = useState("home"); // 'home', 'calculator', 'about', 'documentation'

  // Flowchart States
  const [selectedStrategy, setSelectedStrategy] = useState("conservative");
  const [selectedCreditType, setSelectedCreditType] = useState("consumer");
  const [selectedProduct, setSelectedProduct] = useState("kpr");
  const [userName, setUserName] = useState("Andika");

  // Form input states
  const [nama, setNama] = useState("Wahyu Setiawan");
  const [usia, setUsia] = useState(26);
  const [pekerjaan, setPekerjaan] = useState("Karyawan Swasta");
  const [kota, setKota] = useState("Jakarta");
  const [pendapatan, setPendapatan] = useState(6193292);
  const [cicilan, setCicilan] = useState(2577223);
  const [tunggakan, setTunggakan] = useState(4);
  const [skor, setSkor] = useState(593);
  const [rasio, setRasio] = useState(0.416);

  // Calculated risk state
  const [riskProb, setRiskProb] = useState(0);
  const [apiStatus, setApiStatus] = useState("checking"); // 'online', 'offline', 'checking'
  const [isPending, setIsPending] = useState(false);

  // Sample data (Mock Database) for "tidak sepi" UI
  const mockClients = [
    { nama: "Wahyu Setiawan", pekerjaan: "Karyawan Swasta", pendapatan: 6193292, cicilan: 2577223, rasio: 0.416, tunggakan: 4, skor: 593, usia: 26, is_default: 0 },
    { nama: "Joko Tanjung", pekerjaan: "Pengusaha", pendapatan: 20491471, cicilan: 1208811, rasio: 0.059, tunggakan: 0, skor: 727, usia: 31, is_default: 0 },
    { nama: "Nanda Santoso", pekerjaan: "PNS", pendapatan: 4805010, cicilan: 964941, rasio: 0.201, tunggakan: 3, skor: 652, usia: 47, is_default: 0 },
    { nama: "Lina Suharto", pekerjaan: "Wiraswasta", pendapatan: 3631928, cicilan: 837947, rasio: 0.231, tunggakan: 1, skor: 817, usia: 44, is_default: 1 },
    { nama: "Muhamad Purnama", pekerjaan: "Pengusaha", pendapatan: 18238483, cicilan: 3584894, rasio: 0.197, tunggakan: 0, skor: 742, usia: 40, is_default: 0 },
    { nama: "Irfan Lestari", pekerjaan: "Tidak Bekerja", pendapatan: 1500000, cicilan: 1664100, rasio: 1.109, tunggakan: 0, skor: 690, usia: 50, is_default: 1 }
  ];

  // Dynamic Clients state with localStorage support
  const [clients, setClients] = useState([]);

  // Load clients on mount (Safe client-side check)
  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("crs_clients");
      if (saved) {
        try {
          setClients(JSON.parse(saved));
        } catch (e) {
          setClients(mockClients);
        }
      } else {
        setClients(mockClients);
      }
    }
  }, []);

  // Check backend status on mount
  useEffect(() => {
    const pingBackend = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/`);
        if (res.ok) {
          setApiStatus("online");
        } else {
          setApiStatus("offline");
        }
      } catch (e) {
        setApiStatus("offline");
      }
    };
    pingBackend();
  }, []);

  // Automatically calculate Debt-to-Income (DTI) ratio when income or installment changes
  useEffect(() => {
    if (pendapatan > 0) {
      const calculatedRatio = cicilan / pendapatan;
      setRasio(parseFloat(calculatedRatio.toFixed(3)));
    } else {
      setRasio(0);
    }
  }, [pendapatan, cicilan]);

  // Recalculate model risk when inputs update (tries Backend API, falls back to Local client-side calculation)
  useEffect(() => {
    let active = true;
    const fetchPrediction = async () => {
      setIsPending(true);
      try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            jenis_pekerjaan: pekerjaan,
            pendapatan_per_b: pendapatan,
            rasio_utang_pend: Math.min(rasio, 2.0),
            jumlah_tunggakan: tunggakan,
            usia: usia,
            skor_kredit_sebe: skor
          }),
        });
        
        if (!response.ok) {
          throw new Error("API response error");
        }

        const data = await response.json();
        if (active) {
          setRiskProb(data.risk_probability);
          setApiStatus("online");
        }
      } catch (err) {
        console.warn("Backend API offline or failed, falling back to local client-side inference:", err);
        // Fallback to local JS calculation
        const localProbability = predictDefaultRisk({
          pendapatan,
          rasio,
          tunggakan,
          usia,
          skor,
          pekerjaan
        });
        if (active) {
          setRiskProb(localProbability);
          setApiStatus("offline");
        }
      } finally {
        if (active) {
          setIsPending(false);
        }
      }
    };

    // Debounce the API call slightly to avoid spamming the backend during slide/input changes
    const timeoutId = setTimeout(() => {
      fetchPrediction();
    }, 150);

    return () => {
      active = false;
      clearTimeout(timeoutId);
    };
  }, [pendapatan, rasio, tunggakan, usia, skor, pekerjaan]);

  // Save current calculator client data to state & localStorage
  const handleSaveClient = () => {
    const activeThreshold = METODOLOGI_DATA.products[selectedProduct]?.thresholds[selectedStrategy] || 0.50;
    const isDefault = riskProb >= activeThreshold ? 1 : 0;
    const newClient = {
      nama: nama || "Tanpa Nama",
      pekerjaan,
      pendapatan,
      cicilan,
      rasio,
      tunggakan,
      skor,
      usia,
      is_default: isDefault
    };

    const updated = [newClient, ...clients];
    setClients(updated);
    if (typeof window !== "undefined") {
      localStorage.setItem("crs_clients", JSON.stringify(updated));
    }
    alert(`Data debitur "${newClient.nama}" berhasil disimpan ke tabel database.`);
  };

  // Delete client from state & localStorage
  const handleDeleteClient = (indexToDelete, clientName) => {
    if (confirm(`Apakah Anda yakin ingin menghapus data debitur "${clientName}"?`)) {
      const updated = clients.filter((_, idx) => idx !== indexToDelete);
      setClients(updated);
      if (typeof window !== "undefined") {
        localStorage.setItem("crs_clients", JSON.stringify(updated));
      }
    }
  };

  // Helper to load client into the form
  const loadClient = (client) => {
    setNama(client.nama);
    setUsia(client.usia);
    setPekerjaan(client.pekerjaan);
    setPendapatan(client.pendapatan);
    setCicilan(client.cicilan);
    setTunggakan(client.tunggakan);
    setSkor(client.skor);
    setRasio(client.rasio);
    setCurrentTab("calculator"); // Auto switch to calculator when loading data
  };

  // Helper to format currency to Rupiah
  const formatRupiah = (number) => {
    return new Intl.NumberFormat("id-ID", {
      style: "currency",
      currency: "IDR",
      maximumFractionDigits: 0
    }).format(number);
  };

  const activeThreshold = METODOLOGI_DATA.products[selectedProduct]?.thresholds[selectedStrategy] || 0.50;

  // Determine risk details
  const getRiskDetails = (prob) => {
    if (pendapatan < 2500000) {
      return {
        badgeClass: "risk-badge risk-high",
        color: "#0857C3", // Nusantara Blue
        level: "Tinggi (Danger)",
        desc: "Pendapatan di bawah kriteria minimum Rp2.500.000. Pengajuan kredit otomatis ditolak."
      };
    }
    
    // Low risk: less than activeThreshold - 0.15
    const lowLimit = Math.max(0.15, activeThreshold - 0.15);
    
    if (prob < lowLimit) {
      return {
        badgeClass: "risk-badge risk-low",
        color: "#71C5E8", // Mentari Blue
        level: "Rendah (Safe)",
        desc: `Kolektibilitas diprediksi Lancar (di bawah batas ${lowLimit.toFixed(2)}). Pengajuan kredit aman untuk disetujui.`
      };
    } else if (prob < activeThreshold) {
      return {
        badgeClass: "risk-badge risk-medium",
        color: "#307FE2", // Cakrawala Blue
        level: "Menengah (Caution)",
        desc: `Risiko Sedang. Probabilitas default mendekati batas kritis (${activeThreshold.toFixed(2)}). Diperlukan penelaahan dokumen jaminan tambahan.`
      };
    } else {
      return {
        badgeClass: "risk-badge risk-high",
        color: "#0857C3", // Nusantara Blue
        level: "Tinggi (Danger)",
        desc: `Risiko Default Tinggi (melebihi batas kritis ${activeThreshold.toFixed(2)}). Pengajuan kredit disarankan ditolak.`
      };
    }
  };

  const risk = getRiskDetails(riskProb);

  // SVG Gauge calculations
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (riskProb * circumference);

  return (
    <div className="main-wrapper">
      {/* Navbar Header */}
      <nav className="navbar">
        <div className="navbar-container">
          <div className="logo-container" onClick={() => setCurrentTab("home")} style={{ cursor: "pointer" }}>
            <div className="logo-icon">C</div>
            <span className="logo-text">CrediCheck CRS</span>
          </div>
          <ul className="nav-links">
            <li className={`nav-item ${currentTab === "home" ? "active" : ""}`}>
              <button onClick={() => setCurrentTab("home")}>Beranda</button>
            </li>
            <li className={`nav-item ${currentTab === "calculator" ? "active" : ""}`}>
              <button onClick={() => setCurrentTab("calculator")}>Kalkulator Risiko</button>
            </li>
            <li className={`nav-item ${currentTab === "about" ? "active" : ""}`}>
              <button onClick={() => setCurrentTab("about")}>Tentang Kami</button>
            </li>
            <li className={`nav-item ${currentTab === "documentation" ? "active" : ""}`}>
              <button onClick={() => setCurrentTab("documentation")}>Dokumentasi</button>
            </li>
          </ul>
        </div>
      </nav>

      {/* Main Dashboard Container */}
      <div className="dashboard-container">
        
        {/* ==========================================
            TAB 1: HOMEPAGE (BERANDA)
            ========================================== */}
        {currentTab === "home" && (
          <div>
            <section className="hero-section">
              <div className="hero-content">
                <div className="hero-badge">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" style={{width: '1rem', height: '1rem'}}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 0 1-1.043 3.296 3.745 3.745 0 0 1-3.296 1.043A3.745 3.745 0 0 1 12 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 0 1-3.296-1.043 3.745 3.745 0 0 1-1.043-3.296A3.745 3.745 0 0 1 3 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 0 1 1.043-3.296 3.746 3.746 0 0 1 3.296-1.043A3.746 3.746 0 0 1 12 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 0 1 3.296 1.043 3.746 3.746 0 0 1 1.043 3.296A3.745 3.745 0 0 1 21 12Z" />
                  </svg>
                  Keputusan Kredit Berbasis Machine Learning
                </div>
                <h1>Analisis Risiko Debitur secara Instan dengan <span>CrediCheck CRS</span></h1>
                <p className="hero-description">
                  CrediCheck Credit Risk System (CRS) mengintegrasikan model prediktif berbasis sains data untuk mengevaluasi kelayakan kredit debitur secara real-time. Membantu bank dan lembaga pembiayaan menekan tingkat Non-Performing Loan (NPL) secara efektif.
                </p>
                <div className="hero-buttons">
                  <button className="btn btn-primary" onClick={() => setCurrentTab("calculator")}>
                    Mulai Kalkulator
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{width: '1.15rem', height: '1.15rem'}}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
                    </svg>
                  </button>
                  <button className="btn btn-secondary" onClick={() => setCurrentTab("about")}>Pelajari Metodologi</button>
                </div>
              </div>
            </section>

            {/* Simulation Quiz Section */}
            <div className="card" style={{ marginBottom: '3rem', background: 'linear-gradient(135deg, #FFFFFF 0%, #F1F7FC 100%)', border: '2px solid var(--cakrawala-blue)' }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--nusantara-blue)', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" style={{width: '1.5rem', height: '1.5rem'}}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 5.25h.008v.008H12v-.008Z" />
                </svg>
                Simulasi Konfigurasi Model (Pertanyaan Analis)
              </h2>
              <p style={{ fontSize: '0.9rem', color: '#475569', marginBottom: '1.5rem' }}>
                Jawab pertanyaan berikut untuk menentukan model default system dan batas toleransi risiko (cut-off threshold) yang akan digunakan pada Kalkulator Risiko.
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
                <div className="form-group">
                  <label className="form-label" htmlFor="quiz-username">Nama Anda (Analis / Petugas)</label>
                  <input
                    id="quiz-username"
                    type="text"
                    className="form-input"
                    value={userName}
                    onChange={(e) => setUserName(e.target.value)}
                    placeholder="Masukkan nama Anda..."
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="quiz-strategy">1. Apa strategi bisnis penyaluran kredit saat ini?</label>
                  <select
                    id="quiz-strategy"
                    className="form-input"
                    value={selectedStrategy}
                    onChange={(e) => setSelectedStrategy(e.target.value)}
                  >
                    <option value="conservative">Konservatif (Utamakan NPL Rendah & Pengetatan Batasan)</option>
                    <option value="expansion">Ekspansi (Utamakan Volume Penyaluran & Longgar)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="quiz-credittype">2. Apa jenis kredit yang diajukan oleh calon debitur?</label>
                  <select
                    id="quiz-credittype"
                    className="form-input"
                    value={selectedCreditType}
                    onChange={(e) => {
                      const val = e.target.value;
                      setSelectedCreditType(val);
                      // Auto select first product of that category
                      if (val === 'consumer') setSelectedProduct('kpr');
                      else if (val === 'micro') setSelectedProduct('kur');
                      else if (val === 'small') setSelectedProduct('kmk');
                    }}
                  >
                    <option value="consumer">Consumer (Konsumer)</option>
                    <option value="micro">Mikro (Micro)</option>
                    <option value="small">Small (Ritel/Kecil)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="quiz-product">3. Tentukan produk kredit spesifik:</label>
                  <select
                    id="quiz-product"
                    className="form-input"
                    value={selectedProduct}
                    onChange={(e) => setSelectedProduct(e.target.value)}
                  >
                    {selectedCreditType === 'consumer' && (
                      <>
                        <option value="kpr">KPR (Kredit Pemilikan Rumah)</option>
                        <option value="kkb">KKB (Kredit Kendaraan Bermotor)</option>
                        <option value="kartu_kredit">Kartu Kredit</option>
                        <option value="ceria">Ceria (Digital Paylater)</option>
                        <option value="briguna">Briguna (Kredit Tanpa Agunan)</option>
                      </>
                    )}
                    {selectedCreditType === 'micro' && (
                      <>
                        <option value="kur">KUR (Kredit Usaha Rakyat)</option>
                        <option value="kur_kecil">KUR Kecil (Plafond Tertinggi)</option>
                        <option value="kupedes">Kupedes (Kredit Umum Pedesaan)</option>
                      </>
                    )}
                    {selectedCreditType === 'small' && (
                      <>
                        <option value="kmk">Kredit Modal Kerja Retail</option>
                        <option value="investasi">Kredit Investasi Retail</option>
                      </>
                    )}
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="quiz-city">4. Tentukan Kota Operasional / Wilayah Kredit:</label>
                  <select
                    id="quiz-city"
                    className="form-input"
                    value={kota}
                    onChange={(e) => setKota(e.target.value)}
                  >
                    <option value="Bandung">Bandung</option>
                    <option value="Batam">Batam</option>
                    <option value="Bekasi">Bekasi</option>
                    <option value="Bogor">Bogor</option>
                    <option value="Denpasar">Denpasar</option>
                    <option value="Depok">Depok</option>
                    <option value="Jakarta">Jakarta</option>
                    <option value="Makassar">Makassar</option>
                    <option value="Malang">Malang</option>
                    <option value="Medan">Medan</option>
                    <option value="Palembang">Palembang</option>
                    <option value="Semarang">Semarang</option>
                    <option value="Surabaya">Surabaya</option>
                    <option value="Tangerang">Tangerang</option>
                    <option value="Yogyakarta">Yogyakarta</option>
                  </select>
                </div>
              </div>



              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button className="btn btn-primary" onClick={() => setCurrentTab('calculator')}>
                  Terapkan & Lanjut ke Kalkulator Risiko
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{width: '1.15rem', height: '1.15rem'}}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Feature Cards Grid */}
            <h2 className="section-title">Mengapa Menggunakan CrediCheck CRS?</h2>
            <div className="features-grid">
              <div className="feature-card">
                <div className="feature-icon-wrapper">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{width: '1.5rem', height: '1.5rem'}}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                  </svg>
                </div>
                <h3>Prediksi Risiko Instan</h3>
                <p>Proses analisis hanya memakan waktu kurang dari 150 milidetik berkat pemrosesan model di sisi server dan fallback inference mesin JavaScript internal.</p>
              </div>

              <div className="feature-card">
                <div className="feature-icon-wrapper">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{width: '1.5rem', height: '1.5rem'}}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 0 1-1.043 3.296 3.745 3.745 0 0 1-3.296 1.043A3.745 3.745 0 0 1 12 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 0 1-3.296-1.043 3.745 3.745 0 0 1-1.043-3.296A3.745 3.745 0 0 1 3 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 0 1 1.043-3.296 3.746 3.746 0 0 1 3.296-1.043A3.746 3.746 0 0 1 12 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 0 1 3.296 1.043 3.746 3.746 0 0 1 1.043 3.296A3.745 3.745 0 0 1 21 12Z" />
                  </svg>
                </div>
                <h3>Standardisasi Keputusan</h3>
                <p>Menghilangkan subjektivitas keputusan kredit analis manusia dengan mencocokkan profil data historis serta regulasi batas rasio Debt-to-Income (DTI).</p>
              </div>

              <div className="feature-card">
                <div className="feature-icon-wrapper">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{width: '1.5rem', height: '1.5rem'}}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
                  </svg>
                </div>
                <h3>Simpan & Kelola Simulasi</h3>
                <p>Simpan hasil kalkulasi debitur Anda ke dalam database simulasi lokal. Data dapat dihapus dan dimuat ulang kapan pun dibutuhkan untuk peninjauan.</p>
              </div>
            </div>

            {/* Workflow Timeline */}
            <div className="workflow-section">
              <h2 className="section-title" style={{ marginBottom: '2rem' }}>Alur Kerja Analisis CRS</h2>
              <div className="workflow-steps">
                <div className="workflow-step">
                  <div className="step-number">1</div>
                  <h4>Input Parameter</h4>
                  <p>Masukkan profil keuangan debitur (Pekerjaan, Usia, Pendapatan, Cicilan, Skor Kredit, dan Tunggakan).</p>
                </div>
                <div className="workflow-step">
                  <div className="step-number">2</div>
                  <h4>Inference Engine</h4>
                  <p>Model mengevaluasi data input dengan bobot Logistic Regression terlatih secara real-time.</p>
                </div>
                <div className="workflow-step">
                  <div className="step-number">3</div>
                  <h4>Keputusan & Simpan</h4>
                  <p>Lihat status tingkat risiko (Lancar, Sedang, Bahaya), lalu simpan data debitur ke dalam daftar pemantauan.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==========================================
            TAB 2: RISK CALCULATOR (KALKULATOR RISIKO)
            ========================================== */}
        {currentTab === "calculator" && (
          <div>
            {/* Header info */}
            <div className="header" style={{ marginBottom: '1.5rem' }}>
              <div>
                <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0F172A' }}>Kalkulator Risiko Kredit</h1>
                <p style={{ fontSize: '0.9rem', color: '#64748B', marginTop: '0.25rem' }}>Lakukan prediksi probabilitas default berdasarkan data profil debitur terbaru.</p>
              </div>
              <div className="header-status" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <div 
                    className="status-dot" 
                    style={{ 
                      backgroundColor: apiStatus === 'online' ? '#10B981' : apiStatus === 'checking' ? '#F59E0B' : '#EF4444',
                      animation: apiStatus === 'checking' ? 'pulse 1s infinite' : 'pulse 2s infinite'
                    }}
                  ></div>
                  <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>
                    API: {apiStatus === 'online' ? 'Connected' : apiStatus === 'checking' ? 'Connecting...' : 'Offline (Fallback)'}
                  </span>
                </div>
                <div style={{ width: '1px', height: '14px', backgroundColor: 'var(--border-color)' }}></div>
                <span>Model Aktif (Logistic Regression)</span>
              </div>
            </div>

            {/* Stats Quick Cards */}
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-label">Nama Analis / Petugas</span>
                <span className="stat-value highlight">{userName || "Belum Diatur"}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Produk Kredit Terpilih</span>
                <span className="stat-value" style={{ color: 'var(--cakrawala-blue)' }}>{METODOLOGI_DATA.products[selectedProduct]?.name || "Belum Terpilih"}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Jumlah Debitur Disimpan</span>
                <span className="stat-value">{clients.length}</span>
              </div>
            </div>

            {/* Main Calculator Grid */}
            <div className="dashboard-grid">
              {/* Input Form Card */}
              <div className="card">
                <h2 className="card-title">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{width: '1.25rem', height: '1.25rem'}}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5.586a1 1 0 0 1 .707.293l5.414 5.414a1 1 0 0 1 .293.707V19a2 2 0 0 1-2 2z" />
                  </svg>
                  Kalkulator Risiko Kredit
                </h2>
                
                <form onSubmit={(e) => e.preventDefault()} className="form-grid">
                  <div className="form-group">
                    <label className="form-label" htmlFor="nama">Nama Debitur</label>
                    <input
                      id="nama"
                      type="text"
                      className="form-input"
                      value={nama}
                      onChange={(e) => setNama(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label" htmlFor="usia">Usia (Tahun)</label>
                    <input
                      id="usia"
                      type="number"
                      min="18"
                      max="100"
                      className="form-input"
                      value={usia}
                      onChange={(e) => setUsia(parseInt(e.target.value) || 0)}
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label" htmlFor="pekerjaan">Jenis Pekerjaan</label>
                    <select
                      id="pekerjaan"
                      className="form-input"
                      value={pekerjaan}
                      onChange={(e) => setPekerjaan(e.target.value)}
                    >
                      <option value="Karyawan Swasta">Karyawan Swasta</option>
                      <option value="PNS">PNS</option>
                      <option value="Pengusaha">Pengusaha</option>
                      <option value="Tidak Bekerja">Tidak Bekerja</option>
                      <option value="Wiraswasta">Wiraswasta</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label" htmlFor="tunggakan">Jumlah Tunggakan (Bulan)</label>
                    <input
                      id="tunggakan"
                      type="number"
                      min="0"
                      max="24"
                      className="form-input"
                      value={tunggakan}
                      onChange={(e) => setTunggakan(parseInt(e.target.value) || 0)}
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label" htmlFor="pendapatan">Pendapatan Per Bulan (Rp)</label>
                    <input
                      id="pendapatan"
                      type="number"
                      min="1"
                      className="form-input"
                      value={pendapatan}
                      onChange={(e) => setPendapatan(parseInt(e.target.value) || 0)}
                    />
                    <span style={{fontSize: '0.75rem', color: '#64748B'}}>{formatRupiah(pendapatan)}</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label" htmlFor="cicilan">Cicilan Per Bulan (Rp)</label>
                    <input
                      id="cicilan"
                      type="number"
                      min="0"
                      className="form-input"
                      value={cicilan}
                      onChange={(e) => setCicilan(parseInt(e.target.value) || 0)}
                    />
                    <span style={{fontSize: '0.75rem', color: '#64748B'}}>{formatRupiah(cicilan)}</span>
                  </div>

                  <div className="form-group full-width">
                    <label className="form-label">
                      Skor Kredit Sebelumnya: <span style={{color: 'var(--nusantara-blue)'}}>{skor}</span>
                    </label>
                    <div className="slider-container">
                      <input
                        type="range"
                        min="300"
                        max="850"
                        className="form-slider"
                        value={skor}
                        onChange={(e) => setSkor(parseInt(e.target.value))}
                      />
                      <span className="slider-val">{skor}</span>
                    </div>
                  </div>

                  <div className="form-group full-width" style={{marginTop: '0.5rem', background: '#F8FAFC', padding: '1rem', borderRadius: '8px', border: '1px dashed var(--border-color)'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between'}}>
                      <span style={{fontWeight: 600, fontSize: '0.875rem', color: '#475569'}}>Debt-to-Income (DTI) Ratio Terhitung:</span>
                      <span style={{fontWeight: 700, fontSize: '0.875rem', color: 'var(--nusantara-blue)'}}>{(rasio * 100).toFixed(1)}% ({rasio})</span>
                    </div>
                  </div>
                </form>
              </div>

              {/* Results Card */}
              <div className="card result-card">
                <h2 className="card-title" style={{alignSelf: 'flex-start'}}>
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{width: '1.25rem', height: '1.25rem'}}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6a7.5 7.5 0 1 0 7.5 7.5h-7.5V6z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 10.5H21A7.5 7.5 0 0 0 13.5 3v7.5z" />
                  </svg>
                  Analisis Risiko Kredit
                </h2>

                {/* dynamic system model card */}
                <div style={{ width: '100%', background: '#F8FAFC', padding: '0.85rem 1rem', borderRadius: '8px', border: '1px solid var(--border-color)', marginBottom: '1.25rem', textAlign: 'left' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748B', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.25rem', letterSpacing: '0.05em' }}>Konfigurasi Model Aktif</div>
                  <div style={{ fontSize: '0.85rem', color: '#1E293B', lineHeight: 1.45 }}>
                    👤 Analis: <strong>{userName}</strong> <br />
                    🎯 Batas Kritis: <strong style={{ color: 'var(--nusantara-blue)' }}>{activeThreshold}</strong> (Produk: {METODOLOGI_DATA.products[selectedProduct]?.name})
                  </div>
                </div>

                <div className="gauge-wrapper" style={{ position: 'relative', transition: 'opacity 0.2s', opacity: isPending ? 0.6 : 1 }}>
                  <svg className="gauge-svg" viewBox="0 0 200 200">
                    <circle className="gauge-bg" cx="100" cy="100" r={radius} />
                    <circle
                      className="gauge-fill"
                      cx="100"
                      cy="100"
                      r={radius}
                      stroke={risk.color}
                      strokeDasharray={circumference}
                      strokeDashoffset={strokeDashoffset}
                    />
                  </svg>
                  <div className="gauge-text">
                    <span className="gauge-pct">{(riskProb * 100).toFixed(1)}%</span>
                    <span className="gauge-label">{isPending ? 'LOADING...' : 'RISK RATE'}</span>
                  </div>
                </div>

                <div className={risk.badgeClass}>
                  RISIKO: {risk.level}
                </div>

                <p className="risk-description">
                  {nama ? <strong>{nama}</strong> : "Debitur"} memiliki probabilitas default sebesar <strong>{(riskProb * 100).toFixed(1)}%</strong>. {risk.desc}
                </p>

                {/* Save to database local simulation button */}
                <div className="save-action-wrapper">
                  <button className="btn btn-primary" onClick={handleSaveClient}>
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{width: '1.15rem', height: '1.15rem'}}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0 3 3m-3-3-3 3M6.75 19.5a4.5 4.5 0 0 1-1.41-8.775 5.25 5.25 0 0 1 10.233-2.33 3 3 0 0 1 3.758 3.848A3.752 3.752 0 0 1 18 19.5H6.75Z" />
                    </svg>
                    Simpan Hasil ke Database
                  </button>
                </div>
              </div>
            </div>

            {/* Database/Mock Table Section (tidak sepi) */}
            <section className="db-section card">
              <div className="table-header">
                <div>
                  <h3 className="table-title">Daftar Aplikasi Kredit Terbaru (Database CRS)</h3>
                  <p style={{fontSize: '0.85rem', color: '#64748B', marginTop: '0.25rem'}}>Klik pada baris untuk memuat ulang data ke kalkulator, atau hapus baris simulasi.</p>
                </div>
              </div>

              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Nama Debitur</th>
                      <th>Usia</th>
                      <th>Jenis Pekerjaan</th>
                      <th>Pendapatan</th>
                      <th>Rasio Utang (DTI)</th>
                      <th>Skor Kredit</th>
                      <th>Tunggakan</th>
                      <th>Prediksi Status</th>
                      <th style={{textAlign: 'center'}}>Aksi</th>
                    </tr>
                  </thead>
                  <tbody>
                    {clients.length === 0 ? (
                      <tr>
                        <td colSpan="9" style={{textAlign: 'center', color: '#94A3B8', padding: '2rem'}}>
                          Belum ada debitur yang disimpan. Masukkan data di atas lalu klik &quot;Simpan Hasil ke Database&quot;.
                        </td>
                      </tr>
                    ) : (
                      clients.map((client, index) => (
                        <tr key={index}>
                          <td 
                            style={{fontWeight: 600, color: 'var(--nusantara-blue)'}}
                            onClick={() => loadClient(client)}
                          >
                            {client.nama}
                          </td>
                          <td onClick={() => loadClient(client)}>{client.usia} Th</td>
                          <td onClick={() => loadClient(client)}>{client.pekerjaan}</td>
                          <td onClick={() => loadClient(client)}>{formatRupiah(client.pendapatan)}</td>
                          <td onClick={() => loadClient(client)}>{(client.rasio * 100).toFixed(1)}%</td>
                          <td onClick={() => loadClient(client)}>{client.skor}</td>
                          <td onClick={() => loadClient(client)}>{client.tunggakan} Bln</td>
                          <td onClick={() => loadClient(client)}>
                            <span className={client.is_default === 1 ? "badge badge-default" : "badge badge-nondefault"}>
                              {client.is_default === 1 ? "Default (High Risk)" : "Lancar (Low/Medium)"}
                            </span>
                          </td>
                          <td style={{textAlign: 'center'}}>
                            <button 
                              className="btn btn-danger" 
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteClient(index, client.nama);
                              }}
                            >
                              Hapus
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {/* ==========================================
            TAB 3: ABOUT US (TENTANG KAMI)
            ========================================== */}
        {currentTab === "about" && (
          <section className="about-section">
            <div className="about-info">
              <h2>Tentang CrediCheck CRS</h2>
              <p>
                CrediCheck Credit Risk System (CRS) dikembangkan sebagai solusi modern berbasis AI untuk membantu institusi keuangan memitigasi risiko kredit bermasalah. Dengan memodelkan data historis pinjaman, sistem ini mampu mendeteksi potensi debitur gagal bayar secara objektif dan instan.
              </p>
              
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '1.5rem', marginBottom: '0.75rem', color: 'var(--nusantara-blue)' }}>Metodologi Model</h3>
              <p>
                Model prediktif inti dibangun menggunakan algoritma <strong>Logistic Regression</strong> dengan bobot seimbang (balanced class weights) untuk mengatasi ketidakseimbangan kelas data latih. Model dilatih menggunakan dataset yang dibersihkan dari pencilan (outliers) dan dinormalisasi menggunakan Standard Scaler.
              </p>
              
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '1.5rem', marginBottom: '0.75rem', color: 'var(--nusantara-blue)' }}>Batasan Sistem & Aturan Bisnis</h3>
              <p>
                Aplikasi ini mengombinasikan kalkulasi machine learning dengan aturan bisnis bank konvensional:
              </p>
              <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', color: '#475569', fontSize: '0.95rem', marginBottom: '1.5rem' }}>
                <li><strong>Pendapatan Minimum:</strong> Debitur dengan pendapatan di bawah Rp2.500.000 otomatis diposisikan sebagai Risiko Tinggi (Ditolak).</li>
                <li><strong>Debt-to-Income (DTI) Capping:</strong> Rasio cicilan bulanan dibanding pendapatan dibatasi maksimal 2.0 untuk stabilitas perhitungan, dengan penambahan probabilitas risiko manual jika DTI melebihi 40%.</li>
              </ul>
            </div>

            {/* Model Weight Parameters Dashboard Card */}
            <div className="param-table-card">
              <h3>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{width: '1.25rem', height: '1.25rem'}}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v5.25c0 .621-.504 1.125-1.125 1.125h-2.25A1.125 1.125 0 0 1 3 18.375v-5.25ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v9.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125v-9.75ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v14.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" />
                </svg>
                Bobot Model (Logistic Coefficients)
              </h3>
              <p className="param-desc">
                Koefisien bernilai positif mempertinggi probabilitas default (risiko), sementara koefisien negatif menurunkan probabilitas default.
              </p>
              
              <div className="param-list">
                <div className="param-row">
                  <span className="param-name">Pendapatan Per Bulan</span>
                  <span className="param-val negative">-16.296 (Menurunkan Risiko)</span>
                </div>
                <div className="param-row">
                  <span className="param-name">Pekerjaan: Tidak Bekerja</span>
                  <span className="param-val positive">+4.389 (Meningkatkan Risiko)</span>
                </div>
                <div className="param-row">
                  <span className="param-name">Jumlah Tunggakan</span>
                  <span className="param-val positive">+0.753 (Meningkatkan Risiko)</span>
                </div>
                <div className="param-row">
                  <span className="param-name">Usia Debitur</span>
                  <span className="param-val positive">+0.617 (Meningkatkan Risiko)</span>
                </div>
                <div className="param-row">
                  <span className="param-name">Pekerjaan: Wiraswasta</span>
                  <span className="param-val positive">+0.993 (Meningkatkan Risiko)</span>
                </div>
                <div className="param-row">
                  <span className="param-name">Skor Kredit Sebelumnya</span>
                  <span className="param-val positive">+0.080 (Meningkatkan Risiko*)</span>
                </div>
                <div className="param-row">
                  <span className="param-name">Pekerjaan: PNS</span>
                  <span className="param-val negative">-1.487 (Menurunkan Risiko)</span>
                </div>
                <div className="param-row">
                  <span className="param-name">Pekerjaan: Pengusaha</span>
                  <span className="param-val negative">-0.771 (Menurunkan Risiko)</span>
                </div>
                <div className="param-row">
                  <span className="param-name">Rasio Utang (DTI)</span>
                  <span className="param-val positive">+0.134 (Meningkatkan Risiko)</span>
                </div>
              </div>
              <div style={{fontSize: '0.75rem', color: '#64748B', marginTop: '1.25rem', lineHeight: 1.4}}>
                *Catatan: Skor kredit memiliki koefisien positif kecil karena pengaruh interaksi fitur dan skala data setelah penskalaan Standard Scaler pada dataset latih.
              </div>
            </div>
          </section>
        )}

        {/* ==========================================
            TAB 4: DOCUMENTATION (DOKUMENTASI)
            ========================================== */}
        {currentTab === "documentation" && (
          <div>
            <div className="header" style={{ marginBottom: '1.5rem' }}>
              <div>
                <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0F172A' }}>Dokumentasi Metodologi Model</h1>
                <p style={{ fontSize: '0.9rem', color: '#64748B', marginTop: '0.25rem' }}>
                  Simulasikan bagaimana kombinasi strategi bisnis bank dan produk kredit menentukan konfigurasi model prediktif dan threshold keputusan.
                </p>
              </div>
            </div>

            <div className="doc-grid">
              {/* Flowchart Panel */}
              <div className="flowchart-card">
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--nusantara-blue)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" style={{width: '1.25rem', height: '1.25rem'}}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M18 18.72a9.003 9.003 0 0 0-16.797-2.197m12.797 2.197c.585-.349 1.104-.803 1.5-1.325m-1.5 1.325a9.003 9.003 0 0 0-11.12-11.12M18 18.72A9 9 0 0 0 5.625 3m12.375 15.72V14.25m0 0v-1.125A2.25 2.25 0 0 0 15.75 10.875H13.5L12 9.375H9.75A2.25 2.25 0 0 0 7.5 11.625v1.125M18 14.25H16.5M12 9.375V6.75m0 0a2.25 2.25 0 1 0-4.5 0M12 6.75h3.75a2.25 2.25 0 0 0 2.25-2.25v-.375" />
                  </svg>
                  Bagan Alur Keputusan Model (Klik Node untuk Simulasi)
                </h3>

                <div className="flowchart-container">
                  {/* STEP 1: Bank Strategy */}
                  <div className="flowchart-step-row">
                    <span className="flowchart-step-label">Langkah 1: Tentukan Strategi Bank</span>
                    <div className="flowchart-nodes">
                      <div 
                        className={`flowchart-node ${selectedStrategy === "conservative" ? "active" : ""}`}
                        onClick={() => setSelectedStrategy("conservative")}
                      >
                        <span>Konservatif</span>
                        <span className="flowchart-node-subtitle">Recall Tinggi (NPL Rendah)</span>
                      </div>
                      <div 
                        className={`flowchart-node ${selectedStrategy === "expansion" ? "active" : ""}`}
                        onClick={() => setSelectedStrategy("expansion")}
                      >
                        <span>Ekspansi</span>
                        <span className="flowchart-node-subtitle">Penyaluran Kredit Agresif</span>
                      </div>
                    </div>
                  </div>

                  <div className="flowchart-connector active"></div>

                  {/* STEP 2: Credit Type */}
                  <div className="flowchart-step-row">
                    <span className="flowchart-step-label">Langkah 2: Pilih Jenis Kredit</span>
                    <div className="flowchart-nodes">
                      <div 
                        className={`flowchart-node ${selectedCreditType === "consumer" ? "active" : ""}`}
                        onClick={() => {
                          setSelectedCreditType("consumer");
                          setSelectedProduct("kpr");
                        }}
                      >
                        <span>Consumer</span>
                        <span className="flowchart-node-subtitle">Ritel Perorangan</span>
                      </div>
                      <div 
                        className={`flowchart-node ${selectedCreditType === "micro" ? "active" : ""}`}
                        onClick={() => {
                          setSelectedCreditType("micro");
                          setSelectedProduct("kur");
                        }}
                      >
                        <span>Mikro</span>
                        <span className="flowchart-node-subtitle">UMKM Ultra-Mikro</span>
                      </div>
                      <div 
                        className={`flowchart-node ${selectedCreditType === "small" ? "active" : ""}`}
                        onClick={() => {
                          setSelectedCreditType("small");
                          setSelectedProduct("kmk");
                        }}
                      >
                        <span>Small</span>
                        <span className="flowchart-node-subtitle">Ritel Komersial</span>
                      </div>
                    </div>
                  </div>

                  <div className="flowchart-connector active"></div>

                  {/* STEP 3: Products */}
                  <div className="flowchart-step-row">
                    <span className="flowchart-step-label">Langkah 3: Tentukan Produk Kredit</span>
                    <div className="flowchart-nodes">
                       {selectedCreditType === "consumer" && (
                        <>
                          <div 
                            className={`flowchart-node ${selectedProduct === "kpr" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("kpr")}
                          >
                            <span>KPR</span>
                            <span className="flowchart-node-subtitle">Perumahan</span>
                          </div>
                          <div 
                            className={`flowchart-node ${selectedProduct === "kkb" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("kkb")}
                          >
                            <span>KKB</span>
                            <span className="flowchart-node-subtitle">Kendaraan</span>
                          </div>
                          <div 
                            className={`flowchart-node ${selectedProduct === "kartu_kredit" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("kartu_kredit")}
                          >
                            <span>Kartu Kredit</span>
                            <span className="flowchart-node-subtitle">Limit Kredit</span>
                          </div>
                          <div 
                            className={`flowchart-node ${selectedProduct === "ceria" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("ceria")}
                          >
                            <span>Ceria</span>
                            <span className="flowchart-node-subtitle">Paylater Digital</span>
                          </div>
                          <div 
                            className={`flowchart-node ${selectedProduct === "briguna" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("briguna")}
                          >
                            <span>Briguna</span>
                            <span className="flowchart-node-subtitle">KTA Payroll</span>
                          </div>
                        </>
                      )}

                      {selectedCreditType === "micro" && (
                        <>
                          <div 
                            className={`flowchart-node ${selectedProduct === "kur" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("kur")}
                          >
                            <span>KUR Mikro</span>
                            <span className="flowchart-node-subtitle">Usaha Rakyat</span>
                          </div>
                          <div 
                            className={`flowchart-node ${selectedProduct === "kur_kecil" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("kur_kecil")}
                          >
                            <span>KUR Kecil</span>
                            <span className="flowchart-node-subtitle">Plafond Tertinggi</span>
                          </div>
                          <div 
                            className={`flowchart-node ${selectedProduct === "kupedes" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("kupedes")}
                          >
                            <span>Kupedes</span>
                            <span className="flowchart-node-subtitle">Usaha Pedesaan</span>
                          </div>
                        </>
                      )}

                      {selectedCreditType === "small" && (
                        <>
                          <div 
                            className={`flowchart-node ${selectedProduct === "kmk" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("kmk")}
                          >
                            <span>KMK Retail</span>
                            <span className="flowchart-node-subtitle">Modal Kerja</span>
                          </div>
                          <div 
                            className={`flowchart-node ${selectedProduct === "investasi" ? "active" : ""}`}
                            onClick={() => setSelectedProduct("investasi")}
                          >
                            <span>Kredit Investasi</span>
                            <span className="flowchart-node-subtitle">Barang Modal</span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="flowchart-connector active"></div>

                  {/* STEP 4: City Selection */}
                  <div className="flowchart-step-row">
                    <span className="flowchart-step-label">Langkah 4: Pilih Wilayah/Kota Operasional</span>
                    <div className="flowchart-nodes" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center' }}>
                      {["Bandung", "Batam", "Bekasi", "Bogor", "Denpasar", "Depok", "Jakarta", "Makassar", "Malang", "Medan", "Palembang", "Semarang", "Surabaya", "Tangerang", "Yogyakarta"].map((cityName) => (
                        <div 
                          key={cityName}
                          className={`flowchart-node ${kota === cityName ? "active" : ""}`}
                          onClick={() => setKota(cityName)}
                          style={{ minWidth: '90px', padding: '0.5rem' }}
                        >
                          <span>{cityName}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Preview Panel */}
              <div className="preview-card">
                <h3>
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" style={{width: '1.25rem', height: '1.25rem'}}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.43l-1.003.828c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.43l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
                  </svg>
                  Konfigurasi Parameter Aktif
                </h3>

                <div className="preview-grid">
                  <div className="preview-item">
                    <span className="preview-label">Strategi Aktif</span>
                    <span className="preview-value highlight-blue">
                      {METODOLOGI_DATA.strategies[selectedStrategy].name}
                    </span>
                  </div>
                  <div className="preview-item">
                    <span className="preview-label">Jenis Kredit</span>
                    <span className="preview-value">
                      {METODOLOGI_DATA.creditTypes[selectedCreditType].name}
                    </span>
                  </div>
                  <div className="preview-item">
                    <span className="preview-label">Produk Terpilih</span>
                    <span className="preview-value">
                      {METODOLOGI_DATA.products[selectedProduct]?.name}
                    </span>
                  </div>
                  <div className="preview-item">
                    <span className="preview-label">Kota/Wilayah Terpilih</span>
                    <span className="preview-value">
                      {kota}
                    </span>
                  </div>

                  <div className="preview-item" style={{ flexDirection: 'column', gap: '0.5rem', alignItems: 'stretch' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span className="preview-label">Risk Cut-off Threshold</span>
                      <span className="preview-value highlight-green" style={{ fontSize: '1.1rem', fontWeight: 800 }}>
                        {METODOLOGI_DATA.products[selectedProduct]?.thresholds[selectedStrategy]}
                      </span>
                    </div>
                    <div className="threshold-visualizer">
                      <div className="threshold-fill"></div>
                      <div 
                        className="threshold-marker" 
                        style={{ left: `${(METODOLOGI_DATA.products[selectedProduct]?.thresholds[selectedStrategy] || 0.5) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                <div className="model-explanation-card">
                  <h4>Strategi Bisnis</h4>
                  <p>{METODOLOGI_DATA.strategies[selectedStrategy].desc}</p>
                </div>

                <div className="model-explanation-card">
                  <h4>Aturan Threshold & Cut-off</h4>
                  <p>{METODOLOGI_DATA.strategies[selectedStrategy].cutoffDesc}</p>
                  <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#64748B' }}>
                    *Untuk produk <strong>{METODOLOGI_DATA.products[selectedProduct]?.name}</strong>, batas kritis peluang default diset pada <strong>{METODOLOGI_DATA.products[selectedProduct]?.thresholds[selectedStrategy]}</strong>. Di atas angka ini, sistem otomatis menyatakan &quot;Default/Risiko Tinggi&quot;.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}


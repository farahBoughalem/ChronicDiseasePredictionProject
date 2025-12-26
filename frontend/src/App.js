import './styles/globals.css';
import logo from './logowhite.svg';
import ScrollArrow from "./components/ScrollArrow";
import DiseaseTabs from "./components/diseases/DiseaseTabs";
import heroImage from "./predimage.png";
import Footer from "./components/Footer";

function App() {
  return (
      <div>
          <div className="app gradient-bg">
              <header className="top-header">
                  <a href="/">
                      <img src={logo} alt="logo" className="app-logo"/>
                      <span className="app-name">ChronicX</span>
                  </a>
              </header>
              <section className="hero-tagline">
                  <div>
                      <p className="tagline-strong">Early Chronic Disease Prediction</p>
                      <p className="tagline-soft">Now Just One Click Away!</p>
                      <p className="features">
                          <span>AI-powered</span>
                          <span className="bulletpoint"> • </span>
                          <span>Data-driven</span>
                          <span className="bulletpoint"> • </span>
                          <span>Privacy-first</span>
                      </p>
                      <a href="#content" className="btn btn-primary btn-lg px-4">Get Started</a>
                  </div>
                  <img src={heroImage} alt="Hero" className="hero-image" />
              </section>
              <ScrollArrow href="#content" />
          </div>
          <main id="content" className="page-content">
              <div className="container py-5">
                  <div className="row justify-content-center">
                      <div className="col-12 col-lg-8">
                          <DiseaseTabs />
                      </div>
                  </div>
              </div>
          </main>
          <Footer />
      </div>
  );
}

export default App;

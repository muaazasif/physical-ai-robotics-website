import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/chapter_1_introduction">
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}

const chapters = [
  {
    title: 'Introduction',
    description: 'Overview of humanoid robotics, history, and fundamental concepts',
  },
  {
    title: 'Physical AI',
    description: 'Integration of AI with physical systems and real-world applications',
  },
  {
    title: 'Visualization & Animation',
    description: 'Visualization and animation techniques for humanoid robotics AI',
  },
  {
    title: 'ML & AI Algorithms',
    description: 'Machine learning and AI algorithms specifically for humanoid robotics',
  },
  {
    title: 'Control Architecture',
    description: 'Control architectures and system integration for humanoid robotics',
  },
  {
    title: 'Sensors & Perception',
    description: 'Sensor systems and perception techniques for humanoid robotics',
  },
];

function getChapterPath(title) {
  switch(title) {
    case 'Introduction':
      return 'chapter_1_introduction';
    case 'Physical AI':
      return 'chapter_2_physical_ai';
    case 'Visualization & Animation':
      return 'chapter_3_visualization_animation';
    case 'ML & AI Algorithms':
      return 'chapter_4_ml_algorithms';
    case 'Control Architecture':
      return 'chapter_5_control_architecture';
    case 'Sensors & Perception':
      return 'chapter_6_sensors_perception';
    default:
      return 'intro';
  }
}

function ChapterCard({title, description}) {
  return (
    <div className="col col--4 margin-bottom--lg">
      <div className="card">
        <div className="card__header">
          <h3>{title}</h3>
        </div>
        <div className="card__body">
          <p>{description}</p>
        </div>
        <div className="card__footer">
          <Link className="button button--primary button--block" to={`/docs/${getChapterPath(title)}`}>
            Read Chapter
          </Link>
        </div>
      </div>
    </div>
  );
}

function Footer() {
  const {siteConfig} = useDocusaurusContext();
  
  return (
    <footer className={clsx('footer', styles.footer)}>
      <div className="container">
        <div className="row">
          <div className="col col--4">
            <h4 className={styles.footerTitle}>{siteConfig.title}</h4>
            <p className={styles.footerText}>
              A comprehensive textbook on Physical AI and Humanoid Robotics with practical examples, 
              labs, and ROS2 code implementations.
            </p>
          </div>
          <div className="col col--2">
            <h5 className={styles.footerSectionTitle}>Chapters</h5>
            <ul className={styles.footerLinks}>
              <li><Link to="/docs/chapter_1_introduction">Introduction</Link></li>
              <li><Link to="/docs/chapter_2_physical_ai">Physical AI</Link></li>
              <li><Link to="/docs/chapter_3_visualization_animation">Visualization</Link></li>
              <li><Link to="/docs/chapter_4_ml_algorithms">ML & AI</Link></li>
              <li><Link to="/docs/chapter_5_control_architecture">Control Systems</Link></li>
              <li><Link to="/docs/chapter_6_sensors_perception">Sensors & Perception</Link></li>
            </ul>
          </div>
          <div className="col col--2">
            <h5 className={styles.footerSectionTitle}>Resources</h5>
            <ul className={styles.footerLinks}>
              <li><Link to="/docs/chapter_1_introduction">Getting Started</Link></li>

              

            </ul>
          </div>
          <div className="col col--2">
            <h5 className={styles.footerSectionTitle}>Connect</h5>
            <ul className={styles.footerLinks}>
              <li><Link to="https://github.com/muaazasif/physical-ai-robotics-website" target="_blank" rel="noopener">GitHub</Link></li>
              <li><Link to="https://discord.com" target="_blank" rel="noopener">Discord</Link></li>
              <li><Link to="https://twitter.com" target="_blank" rel="noopener">Twitter</Link></li>

            </ul>
          </div>
        </div>
        <div className={styles.footerBottom}>
          <div className="row">
            <div className="col col--6">
              <p className={styles.copyright}>
                Copyright © {new Date().getFullYear()} {siteConfig.title}. Built with Docusaurus.
              </p>
            </div>
            <div className="col col--6">
              <div className={styles.footerSocial}>
                <Link to="https://github.com" className={styles.socialLink} target="_blank" rel="noopener">
                  GitHub
                </Link>
                <span className={styles.socialSeparator}> | </span>
                <Link to="https://twitter.com" className={styles.socialLink} target="_blank" rel="noopener">
                  Twitter
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="A complete textbook on Physical AI & Humanoid Robotics with labs, examples & ROS2 code">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <h2 className={styles.sectionTitle}>Complete Textbook</h2>
                <p className={styles.sectionSubtitle}>
                  This comprehensive textbook covers all aspects of humanoid robotics with practical examples, 
                  hands-on labs, and ROS2 code implementations.
                </p>
              </div>
            </div>
            <div className="row">
              {chapters.map((chapter, idx) => (
                <ChapterCard key={idx} {...chapter} />
              ))}
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </Layout>
  );
}
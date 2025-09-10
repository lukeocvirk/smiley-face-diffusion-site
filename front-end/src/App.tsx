import BackwardDiffusionDemo from './components/BackwardDiffusionDemo'

export default function App() {
  return (
    <div>
      <header className="hero">
        <div className="container inner">
          <h4 className="subtitle">My work at IBM has been confidential, thus this website will instead be on a topic in Computer Science I find interesting. </h4>
          <h1 className="title">Diffusion Models - Summer Work Term Report</h1>
          <h3 className="name">Luke Ocvirk</h3>
          <p className="subtitle">This website introduces the user to diffusion models. First, I will explain the theory and math behind diffusion models, and then you will be able to try it out yourself by viewing a live, scrubbable example I made of a simple 2D diffusion model.</p>
        </div>
      </header>

      <main className="container">
        <div className="grid" style={{ marginBottom: 16 }}>
          <section className="card">
            <h2 className="section-title">What are Diffusion Models?</h2>
            <p>In machine learning, diffusion models are a type of model that learns to create meaningful information from random noise.
              Although diffusion models can be applied to many modalities, the primary use of diffusion models is in image generation.
              <br></br><br></br>Diffusion models are trained via a process in which they learn to "denoise" images while random noise is added gradually.
              This process involves two primary elements: forward diffusion and the reverse process.
              <br></br><br></br>Forward diffusion involves taking an image x<sub>0</sub> and gradually adding Gaussian noise over the course of <i>T</i> steps until it becomes completely random noise.
              <br></br><br></br>The reverse process involves reversing the forward diffusion.
              At each step of the reverse process, the diffusion model attempts to remove the random noise added by the forward diffusion process.
              <br></br><br></br>By training a model this way over the course of thousands or even millions and billions of image samples,
              the model can eventually learn how to produce a meaningful image by denoising an image that originally had only random noise.
              <br></br><br></br>(Weng, 2021)
              </p>
          </section>
          <section className="card">
            <h2 className="section-title">Forward Diffusion</h2>
            <p>Given a noise-free image, x<sub>0</sub>, forward diffusion involves adding Gaussian noise at each step <i>t</i>, according to the variance schedule <i>β</i><sub>t</sub>.
            In doing so, a sequence emerges defined as x<sub>0</sub> → x<sub>1</sub> → … → x<sub><i>T</i></sub> where <i>T</i> is predominantly random noise.
            <br></br><br></br>This forward process is defined as:
              <div style={{ display: 'flex', justifyContent: 'center', margin: '16px 0' }}>
                <img src="./assets/forward_single.png" style={{ width: '100%', maxWidth: '362px' }} />
              </div>
            <br></br>Compounding over multiple steps to:
              <div style={{ display: 'flex', justifyContent: 'center', margin: '16px 0' }}>
                <img src="./assets/forward_multi.png" style={{ width: '100%', maxWidth: '300px' }} />
              </div>
            <div style={{ display: 'flex', justifyContent: 'center', margin: '16px 0' }}>
              <img src="./assets/forward-diffusion.png" style={{ width: '100%', maxWidth: '750px' }} />
            </div>
            <br></br><br></br>(Karagiannokos & Adaloglou, 2022)
            </p>
          </section>
          <section className="card">
            <h2 className="section-title">Reverse Process</h2>
            <p>Given a noisy sample x<sub>t</sub>, the model is trained how to denoise it.
            <br></br><br></br>This process is defined by the model learning to predict what noise was added by the forward diffusion process between the image it is seeing and the image at the previous step.
            <br></br><br></br>For a more concrete explanation of this process, imagine a function which, given an image, could compute how likely the image is to be within the space of natural/real images in pixel/image state space.
            Through denoising, the diffusion model learns to estimate the gradient of this function which, the reverse process leverages to transition a pure noise sample into the distribution of natural images.
            <br></br><br></br>The model is given a large dataset of images that it attempts to denoise slightly using this function; in effect, learning how to denoise data it is given by a small amount.
            <br></br><br></br>This reverse process is defined as:
              <div style={{ display: 'flex', justifyContent: 'center', margin: '16px 0' }}>
                <img src="./assets/backward_single.png" style={{ width: '100%', maxWidth: '420px' }} />
              </div>
            <br></br>Compounding over multiple steps to:
              <div style={{ display: 'flex', justifyContent: 'center', margin: '16px 0' }}>
                <img src="./assets/backward_multi.png" style={{ width: '100%', maxWidth: '346px' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'center', margin: '16px 0' }}>
                <img src="./assets/reverse-process.png" style={{ width: '100%', maxWidth: '750px' }} />
              </div>
              <br></br><br></br>(Karagiannokos & Adaloglou, 2022)
            </p>
          </section>
          <section className="card">
            <h2 className="section-title">Sampling (Image Generation)</h2>
            <p>The sampling or generation stage leverages the denoising capability learned during the reverse process stage.
              <br></br><br></br>At this point, rather than taking any image with some amount of noise and then attempting to denoise it slightly,
              we take an image with complete noise and then run the learned denoising (reverse) process on it repeatedly, perhaps thousands of times until
              we reach <i>t</i> = 0. At this point, if the model was trained correctly the completely noisy image should have been transformed into
              a meaningful image (depending on the data on which it was trained).
              <br></br><br></br>(Karagiannokos & Adaloglou, 2022)
            </p>
          </section>
        </div>

        <BackwardDiffusionDemo />
      </main>

      <footer className="hero">
        <div className="container">
          <h2 className="section-title">Sources</h2>
          <ul className="sources-list">
            <li><a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/" target="_blank" rel="noopener noreferrer">What are Diffusion Models? (Weng, 2021)</a></li>
            <li><a href="https://theaisummer.com/diffusion-models/" target="_blank" rel="noopener noreferrer">How diffusion models work: the math from scratch (Karagiannokos & Adaloglou, 2022)</a></li>
          </ul>
        </div>
      </footer>

    </div>
  )
}

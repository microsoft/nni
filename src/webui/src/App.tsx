import * as React from 'react';
import './App.css';
import SlideBar from './components/SlideBar';

class App extends React.Component<{}, {}> {  
  render () {
    return (
      <div className="App">
        {/* <header className="header_title"><img src={require('./logo.jpg')} alt=""/></header> */}
        <header className="header_title">Neural Network Intelligence</header>
        <div className="content">
          <SlideBar />
          <div className="right">{this.props.children}</div>
        </div>
      </div>
    );
  }
}

export default App;

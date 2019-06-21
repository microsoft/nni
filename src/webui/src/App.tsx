import * as React from 'react';
import { Row, Col } from 'antd';
import './App.css';
import SlideBar from './components/SlideBar';

interface AppState {
  interval: number;
  whichPageToFresh: string;
}

class App extends React.Component<{}, AppState> {
  public _isMounted: boolean;
  constructor(props: {}) {
    super(props);
    this.state = {
      interval: 10, // sendons
      whichPageToFresh: ''
    };
  }

  changeInterval = (interval: number) => {
    if (this._isMounted === true) {
      this.setState(() => ({ interval: interval }));
    }
  }

  changeFresh = (fresh: string) => {
    // interval * 1000 
    if (this._isMounted === true) {
      this.setState(() => ({ whichPageToFresh: fresh }));
    }
  }

  componentDidMount() {
    this._isMounted = true;
  }

  componentWillUnmount() {
    this._isMounted = false;
  }
  render() {
    const { interval, whichPageToFresh } = this.state;
    const reactPropsChildren = React.Children.map(this.props.children, child =>
      // tslint:disable-next-line:no-any
      React.cloneElement(child as React.ReactElement<any>, { interval, whichPageToFresh })
    );
    return (
      <Row className="nni" style={{ minHeight: window.innerHeight }}>
        <Row className="header">
          <Col span={1} />
          <Col className="headerCon" span={22}>
            <SlideBar changeInterval={this.changeInterval} changeFresh={this.changeFresh}/>
          </Col>
          <Col span={1} />
        </Row>
        <Row className="contentBox">
          <Row className="content">
            {reactPropsChildren}
          </Row>
        </Row>
      </Row>
    );
  }
}

export default App;

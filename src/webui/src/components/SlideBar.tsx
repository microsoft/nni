import * as React from 'react';
import { Link } from 'react-router';
import axios from 'axios';
import { Row, Col, Button } from 'antd';
import { MANAGER_IP } from '../static/const';
import '../static/style/slideBar.scss';
import '../static/style/button.scss';

interface SliderState {
  downBool: boolean;
  version: string;
}

class SlideBar extends React.Component<{}, SliderState> {

  public _isMounted = false;

  constructor(props: {}) {
    super(props);
    this.state = {
      downBool: false,
      version: ''
    };
  }

  downExperimentContent = () => {
    this.setState(() => ({
      downBool: true
    }));
    axios
      .all([
        axios.get(`${MANAGER_IP}/experiment`),
        axios.get(`${MANAGER_IP}/trial-jobs`),
        axios.get(`${MANAGER_IP}/metric-data`)
      ])
      .then(axios.spread((res, res1, res2) => {
        if (res.status === 200 && res1.status === 200 && res2.status === 200) {
          if (res.data.params.searchSpace) {
            res.data.params.searchSpace = JSON.parse(res.data.params.searchSpace);
          }
          const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
          const interResultList = res2.data;
          const contentOfExperiment = JSON.stringify(res.data, null, 2);
          let trialMessagesArr = res1.data;
          Object.keys(trialMessagesArr).map(item => {
            // transform hyperparameters as object to show elegantly
            trialMessagesArr[item].hyperParameters = JSON.parse(trialMessagesArr[item].hyperParameters);
            const trialId = trialMessagesArr[item].id;
            // add intermediate result message
            trialMessagesArr[item].intermediate = [];
            Object.keys(interResultList).map(key => {
              const interId = interResultList[key].trialJobId;
              if (trialId === interId) {
                trialMessagesArr[item].intermediate.push(interResultList[key]);
              }
            });
          });
          const trialMessages = JSON.stringify(trialMessagesArr, null, 2);
          const aTag = document.createElement('a');
          const file = new Blob([contentOfExperiment, trialMessages], { type: 'application/json' });
          aTag.download = 'experiment.json';
          aTag.href = URL.createObjectURL(file);
          aTag.click();
          if (!isEdge) {
            URL.revokeObjectURL(aTag.href);
          }
          if (navigator.userAgent.indexOf('Firefox') > -1) {
            const downTag = document.createElement('a');
            downTag.addEventListener('click', function () {
              downTag.download = 'experiment.json';
              downTag.href = URL.createObjectURL(file);
            });
            let eventMouse = document.createEvent('MouseEvents');
            eventMouse.initEvent('click', false, false);
            downTag.dispatchEvent(eventMouse);
          }
          this.setState(() => ({
            downBool: false
          }));
        }
      }));
  }

  getNNIversion = () => {
    axios(`${MANAGER_IP}/version`, {
      method: 'GET'
    })
      .then(res => {
        if (res.status === 200 && this._isMounted) {
          this.setState({ version: res.data });
        }
      });
  }

  componentDidMount() {
    this._isMounted = true;
    this.getNNIversion();
  }

  componentWillUnmount() {
    this._isMounted = false;
  }
  render() {
    const { downBool, version } = this.state;
    return (
      <Row className="nav">
        <Col span={8}>
          <ul className="link">
            <li className="logo">
              <Link to={'/oview'}>
                <img src={require('../static/img/logo2.png')} style={{ width: 88 }} alt="NNI logo" />
              </Link>
            </li>
            <li className="tab firstTab">
              <Link to={'/oview'} activeClassName="high">
                Overview
            </Link>
            </li>
            <li className="tab">
              <Link to={'/detail'} activeClassName="high">
                Trials Detail
            </Link>
            </li>
          </ul>
        </Col>
        <Col span={16} className="feedback">
          <Button
            type="primary"
            className="changeBtu download"
            onClick={this.downExperimentContent}
            disabled={downBool}
          >
            <img src={require('../static/img/icon/download.png')} alt="icon" />
            <span>Download</span>
          </Button>
          <a href="https://github.com/Microsoft/nni/issues/new" target="_blank">
            <img
              src={require('../static/img/icon/issue.png')}
              alt="NNI github issue"
            />
            FeedBack
          </a>
          <span className="version">Version: {version}</span>
        </Col>

      </Row>
    );
  }
}

export default SlideBar;
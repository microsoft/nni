import * as React from 'react';
import MonacoEditor from 'react-monaco-editor';
import { MONACO } from '../../static/const';

interface SearchspaceProps {
    searchSpace: object;
}

class SearchSpace extends React.Component<SearchspaceProps, {}> {
    constructor(props: SearchspaceProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { searchSpace } = this.props;
        return (
            <div className='searchSpace'>
                <MonacoEditor
                    height='361'
                    language='json'
                    theme='vs-light'
                    value={JSON.stringify(searchSpace, null, 2)}
                    options={MONACO}
                />
            </div>
        );
    }
}

export default SearchSpace;

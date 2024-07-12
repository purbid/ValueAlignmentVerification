<h1>
    Value Alignment Verification
</h1>
<h4>
    Rishanth Rajendhran
</h4>
<p>
    This is the repository for the research under the supervision of Prof. Ana Marasovic and Prof. Daniel Brown at The University of Utah on automatic tests to verify alignment of LLMs with human-defined values.
</p>
<h5>
    Setup
</h5>
<p>
    Data and RLHF setup from <a href="https://github.com/allenai/FineGrainedRLHF">allenai/FineGrainedRLHF</a>
</p>
<h5>
    Files
</h5>
<ul>
    <li>
        <h5>
            processData.py
        <h5>
        <p>
            This file is used to transform data from source to the format expected in the VAV setup. 
        </p>
    </li>
    <li>
        <h5>
            processData.py
        <h5>
        <p>
            This file is used to transform data from source to the format expected in the VAV setup. 
        </p>
    </li>
    <li>
        <h5>
            removeRedundancy.py
        <h5>
        <p>
            This file is used to find redundant preference pairs in dataset formatted using processData.py 
        </p>
    </li>
    <li>
        <h5>
            train.py
        <h5>
        <p>
            This file is used to train a model on a dataset (Grammaticality/Fluency)
        </p>
    </li>
    <li>
        <h5>
            test.py
        <h5>
        <p>
            This file is used to test models on redundant and non-redundant preference pairs obtained from removeRedundancy.py 
        </p>
    </li>
    <li>
        <h5>
            monteCarloTest.py
        <h5>
        <p>
            This file is used to check if a randomly sampled point in the weight vector space is any more likely to be inside the intersection of half-spaces spanned by the non-redundant preference pairs obtained from removeRedundancy.py 
        </p>
    </li>
</ul>